import torch
import pickle

import dynaphos
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from dynaphos.utils import get_data_kwargs

from typing import Tuple

import model

from torch.utils.data import Subset

import local_datasets
from torch.utils.data import DataLoader
from utils import resize, tensor_to_rgb, undo_standardize, dilation3x3, CustomSummaryTracker
import torch.nn as nn
import torch.nn.functional as F

import wandb

class LossTerm():
    """Loss term that can be used for the compound loss"""

    def __init__(self, name=None, func=torch.nn.functional.mse_loss, arg_names=None, weight=1.):
        self.name = name
        self.func = func  # the loss function
        self.arg_names = arg_names  # the names of the inputs to the loss function
        self.weight = weight  # the relative weight of the loss term


class CompoundLoss():
    """Helper class for combining multiple loss terms. Initialize with list of
    LossTerm instances. Returns dict with loss terms and total loss"""

    def __init__(self, loss_terms):
        self.loss_terms = loss_terms

    def __call__(self, loss_targets):
        """Calculate all loss terms and the weighted sum"""
        self.out = dict()
        self.out['total'] = 0
        for lt in self.loss_terms:
            func_args = [loss_targets[name] for name in lt.arg_names]  # Find the loss targets by their name
            self.out[lt.name] = lt.func(*func_args)  # calculate result and add to output dict
            self.out['total'] += self.out[lt.name] * lt.weight  # add the weighted loss term to the total
        return self.out

    def items(self):
        """return dict with loss tensors as dict with Python scalars"""
        return {k: v.item() for k, v in self.out.items()}


class RunningLoss():
    """Helper class to track the running loss over multiple batches."""

    def __init__(self):
        self.dict = dict()
        self.reset()

    def reset(self):
        self._counter = 0
        for key in self.dict.keys():
            self.dict[key] = 0.

    def update(self, new_entries):
        """Add the current loss values to the running loss"""
        self._counter += 1
        for key, value in new_entries.items():
            if key in self.dict:
                self.dict[key] += value
            else:
                self.dict[key] = value

    def get(self):
        """Get the average loss values (total loss dived by the processed batch count)"""
        out = {key: (value / self._counter) for key, value in self.dict.items()}
        return out


class L1FeatureLoss(object):
    def __init__(self):
        self.feature_extractor = model.VGGFeatureExtractor(device=torch.device)
        self.loss_fn = torch.nn.functional.l1_loss

    def __call__(self, y_pred, y_true, ):
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)
        err = [self.loss_fn(pred, true) for pred, true in zip(pred_features, true_features)]
        return torch.mean(torch.stack(err))

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.nn.functional.softmax(prediction, dim=1)
        target = torch.nn.functional.one_hot(target, num_classes=prediction.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (prediction * target).sum(dim=(2, 3))
        union = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()

    
def get_dataset(cfg):
    if cfg['dataset'] == 'ADE20K':
        trainset, valset = local_datasets.get_ade20k_dataset(cfg)
    elif cfg['dataset'] == 'BouncingMNIST':
        trainset, valset = local_datasets.get_bouncing_mnist_dataset(cfg)
    elif cfg['dataset'] == 'Characters':
        trainset, valset = local_datasets.get_character_dataset(cfg)
    elif cfg['dataset'] == 'LaPa':
        trainset, valset = local_datasets.get_lapa_dataset(cfg)
    
    if cfg['circular_mask'] is not False:
        cfg['circular_mask'] = trainset._mask.to(cfg['device'])
        
    if cfg['debug_subset']:
        # Subset for debugging: Use only the first 100 samples from each dataset
        num_samples_debug = cfg['debug_subset']  # Define how many samples you want to use for debugging
        trainset = Subset(trainset, indices=range(min(num_samples_debug, len(trainset))))
        valset = Subset(valset, indices=range(min(num_samples_debug, len(valset))))

    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=cfg['batch_size'],shuffle=False, drop_last=True)
    example_batch = next(iter(valloader))

    dataset = {'trainset': trainset,
               'valset': valset,
               'trainloader': trainloader,
               'valloader': valloader,
               'example_batch': example_batch}

    return dataset


def get_models(cfg):
    if cfg['model_architecture'] == 'end-to-end-autoencoder':
        encoder, decoder = model.get_e2e_autoencoder(cfg)
    elif cfg['model_architecture'] == 'end-to-end-autoencoder-nophosphenes':
        encoder, decoder = model.get_e2e_autoencoder_nophosphenes(cfg)
    elif cfg['model_architecture'] == 'zhao-autoencoder':
        encoder, decoder = model.get_Zhao_autoencoder(cfg)
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.get('lr_factor', 0.1), patience=cfg.get('lr_patience', 10), verbose=True)

    if cfg['model_architecture'] == 'end-to-end-autoencoder-nophosphenes':
        simulator = None
    else:
        simulator = get_simulator(cfg)

    models = {'encoder' : encoder,
              'decoder' : decoder,
              'optimizer': optimizer,
              'scheduler': scheduler,
              'simulator': simulator,}
    
    # added section for exp3 with interaction layer
    if 'interaction' in cfg.keys(): 
        with open(cfg['electrode_coords'], 'rb') as handle:
            electrode_coords = pickle.load(handle)
        models['interaction'] = model.get_interaction_model(electrode_coords, simulator.data_kwargs, cfg['interaction'])

    return models


def get_simulator(cfg):
    # initialise simulator
    params = dynaphos.utils.load_params(cfg['base_config'])
    params['run'].update(cfg)
    params['thresholding'].update(cfg)
    device = get_data_kwargs(params)['device']

    with open(cfg['phosphene_map'], 'rb') as handle:
        coordinates_visual_field = pickle.load(handle, )
    simulator = PhospheneSimulator(params, coordinates_visual_field)
    cfg['SPVsize'] = simulator.phosphene_maps.shape[-2:]
    return simulator


def get_logging(cfg):
    out = dict()
        
    wandb.init(project=cfg['project_name'], entity=cfg['entity'], name=cfg['run_name'], config=cfg)
    out['logger'] = wandb
    out['training_loss'] = RunningLoss()
    out['validation_loss'] = RunningLoss()
    out['training_summary'] = CustomSummaryTracker()
    out['validation_summary'] = CustomSummaryTracker()
    out['example_output'] = CustomSummaryTracker()
    return out

####### ADJUST OR ADD TRAINING PIPELINE BELOW

def get_training_pipeline(cfg):
    if cfg['pipeline'] == 'unsupervised-segmentation':
        forward, lossfunc = get_pipeline_unsupervised_segmentation(cfg)
    if cfg['pipeline'] == 'supervised-segmentation':
        forward, lossfunc = get_pipeline_supervised_segmentation(cfg)
    elif cfg['pipeline'] == 'segmentation-latent':
        forward, lossfunc = get_pipeline_segmentation_latent(cfg)
    else:
        print(cfg['pipeline'] + 'not supported yet')
        raise NotImplementedError

    return {'forward': forward, 'compound_loss_func': lossfunc}

def get_pipeline_unsupervised_segmentation(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, label = batch
        label_rgb = tensor_to_rgb(label, cfg['num_classes'])

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes) # * cfg['circular_mask']
        reconstruction_rgb = tensor_to_rgb(torch.nn.functional.softmax(reconstruction.detach(), dim=1).argmax(1, keepdims=True), cfg['num_classes'])

        # Output dictionary
        out = {'input': image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction,
               'target': (label * cfg['circular_mask']).squeeze(1),
               'target_resized': resize(label.float() * cfg['circular_mask'], cfg['SPVsize'],),
               'label_rgb': label_rgb,
               'reconstruction_rgb': reconstruction_rgb,
               }


        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    cross_entropy_loss = LossTerm(name='cross_entropy_loss',
                          func=torch.nn.CrossEntropyLoss(weight=torch.tensor(cfg['class_weights'])).to(cfg['device']),
                          arg_names=('reconstruction', 'target'),
                          weight=cfg['cross_entropy_loss_weight'])
    
    dice_loss = LossTerm(name='dice_loss',
                        func=DiceLoss().to(cfg['device']),
                        arg_names=('reconstruction', 'target'),
                        weight=cfg['dice_loss_weight'])
    
    loss_func = CompoundLoss([cross_entropy_loss, dice_loss])

    return forward, loss_func

def get_pipeline_supervised_segmentation(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, label = batch
        label_rgb = tensor_to_rgb(label, cfg['num_classes'])
        unstandardized_image = undo_standardize(image).mean(1, keepdims=True)

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes) # * cfg['circular_mask']
        reconstruction_rgb = tensor_to_rgb(torch.nn.functional.softmax(reconstruction.detach(), dim=1).argmax(1, keepdims=True), cfg['num_classes'])

        # Output dictionary
        out = {'input': image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction,
               'target': (label * cfg['circular_mask']).squeeze(1),
               'target_resized': resize(label.float() * cfg['circular_mask'], cfg['SPVsize'],),
               'label_rgb': label_rgb,
               'reconstruction_rgb': reconstruction_rgb,
               'input_resized': resize(unstandardized_image * cfg['circular_mask'], cfg['SPVsize'])}

        # Sample phosphenes and target at the centers of the phosphenes
        out.update({'phosphene_centers': simulator.sample_centers(phosphenes),
                    'input_centers': simulator.sample_centers(out['input_resized']),
                    'target_centers': simulator.sample_centers(out['target_resized'])})

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    cross_entropy_loss = LossTerm(name='cross_entropy_loss',
                          func=torch.nn.CrossEntropyLoss(weight=torch.tensor(cfg['class_weights'])).to(cfg['device']),
                          arg_names=('reconstruction', 'target'),
                          weight=cfg['cross_entropy_loss_weight'])
    
    dice_loss = LossTerm(name='dice_loss',
                        func=DiceLoss().to(cfg['device']),
                        arg_names=('reconstruction', 'target'),
                        weight=cfg['dice_loss_weight'])
    
    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_centers', 'input_centers'),
                          weight=cfg['regularization_weight'])
    
    loss_func = CompoundLoss([cross_entropy_loss, dice_loss, regul_loss])

    return forward, loss_func

def get_pipeline_segmentation_latent(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']

        # Data manipulation
        image, label = batch
        label_rgb = tensor_to_rgb(label, cfg['num_classes'])

        # Forward pass
        latent = encoder(image)
        reconstruction = decoder(latent)
        if cfg['circular_mask'] is not False:
            reconstruction = reconstruction * cfg['circular_mask']

        reconstruction_rgb = tensor_to_rgb(torch.nn.functional.softmax(reconstruction.detach(), dim=1).argmax(1, keepdims=True), cfg['num_classes'])

        # Output dictionary
        out = {'input': image,
               'latent': latent,
               'reconstruction': reconstruction,
               'target': label.squeeze(1),
               'label_rgb': label_rgb,
               'reconstruction_rgb': reconstruction_rgb}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    cross_entropy_loss = LossTerm(name='cross_entropy_loss',
                          func=torch.nn.CrossEntropyLoss(weight=torch.tensor(cfg['class_weights'])).to(cfg['device']),
                          arg_names=('reconstruction', 'target'),
                          weight=cfg['cross_entropy_loss_weight'])
    
    dice_loss = LossTerm(name='dice_loss',
                        func=DiceLoss().to(cfg['device']),
                        arg_names=('reconstruction', 'target'),
                        weight=cfg['dice_loss_weight'])
    

    loss_func = CompoundLoss([cross_entropy_loss, dice_loss])

    return forward, loss_func