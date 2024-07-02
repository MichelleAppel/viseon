import pickle

import dynaphos
import local_datasets
import model
import torch
import wandb
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from losses import CompoundLoss, DiceLoss, LossTerm, RunningLoss
from torch.utils.data import DataLoader, Subset
from utils import (
    CustomSummaryTracker,
    dilation3x3,
    resize,
    tensor_to_rgb,
    undo_standardize,
)


def get_dataset(cfg):
    if cfg['dataset'] == 'LaPa':
        trainset, valset = local_datasets.get_lapa_dataset(cfg)
    else:
        raise NotImplementedError
    
    if cfg['circular_mask'] is not False:
        cfg['circular_mask'] = trainset._mask.to(cfg['device'])
        
    if cfg['debug_subset']:
        # Subset for debugging: Use only the first 100 samples from each dataset
        num_samples_debug = cfg['debug_subset']  # Define how many samples you want to use for debugging
        trainset = Subset(trainset, indices=range(min(num_samples_debug, len(trainset))))
        valset = Subset(valset, indices=range(min(num_samples_debug, len(valset))))

    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, drop_last=True)
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
    elif cfg['model_architecture'] == 'bio-autoencoder':
        encoder, decoder = model.get_bio_autoencoder(cfg)
    elif cfg['model_architecture'] == 'bio-autoencoder-nophosphenes':
        encoder, decoder = model.get_bio_autoencoder_nophosphenes(cfg)
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.get('lr_factor', 0.1), patience=cfg.get('lr_patience', 10))

    if 'base_config' in cfg.keys():
        simulator = get_simulator(cfg)
    else:
        simulator = None

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
    elif cfg['pipeline'] == 'supervised-segmentation':
        forward, lossfunc = get_pipeline_supervised_segmentation(cfg)
    elif cfg['pipeline'] == 'segmentation-latent':
        forward, lossfunc = get_pipeline_segmentation_latent(cfg)
    elif cfg['pipeline'] == 'boundary-latent':
        forward, lossfunc = get_pipeline_boundary_latent(cfg)
    elif cfg['pipeline'] == 'unsupervised-boundary':
        forward, lossfunc = get_pipeline_unsupervised_boundary(cfg)
    elif cfg['pipeline'] == 'supervised-boundary':
        forward, lossfunc = get_pipeline_supervised_boundary(cfg)
    elif cfg['pipeline'] == 'boundary-no-decoder':
        forward, lossfunc = get_pipeline_boundary_no_decoder(cfg)
    elif cfg['pipeline'] == 'boundary-supervised-segmentation':
        forward, lossfunc = get_pipeline_boundary_supervised_segmentation(cfg)
    else:
        print(cfg['pipeline'] + 'not supported yet')
        raise NotImplementedError

    return {'forward': forward, 'compound_loss_func': lossfunc}

##### SEGMENTATION PIPELINES

def get_pipeline_segmentation_latent(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']

        # Data manipulation
        image, label = batch['image'] , batch['segmentation_maps']
        label_rgb = tensor_to_rgb(label, cfg['num_classes'])

        # Forward pass
        latent = encoder(image)
        reconstruction = decoder(latent)
        if cfg['circular_mask'] is not False:
            reconstruction = reconstruction * cfg['circular_mask']

        reconstruction_rgb = tensor_to_rgb(torch.nn.functional.softmax(reconstruction.detach(), dim=1).argmax(1, keepdims=True), cfg['num_classes'])

        # Output dictionary
        out = {'input': image,
               'phosphenes': latent,
               'reconstruction': reconstruction,
               'target': label.squeeze(1),
               'target_rgb': label_rgb,
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

def get_pipeline_unsupervised_segmentation(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, label = batch['image'] , batch['segmentation_maps']
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
               'target_rgb': label_rgb,
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
        image, label = batch['image'] , batch['segmentation_maps']
        label_rgb = tensor_to_rgb(label, cfg['num_classes'])
        unstandardized_image = image.mean(1, keepdims=True)/255.0

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes) # * cfg['circular_mask']
        reconstruction_rgb = tensor_to_rgb(torch.nn.functional.softmax(reconstruction.detach(), dim=1).argmax(1, keepdims=True), cfg['num_classes'])

        input_resized = resize(unstandardized_image * cfg['circular_mask'], cfg['SPVsize'])

        # Output dictionary
        out = {'input': image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction,
               'target': (label * cfg['circular_mask']).squeeze(1),
               'target_resized': input_resized,
               'target_rgb': label_rgb,
               'reconstruction_rgb': reconstruction_rgb,
               'input_resized': input_resized}

        # Sample phosphenes and target at the centers of the phosphenes
        out.update({'phosphene_centers': simulator.sample_centers(phosphenes),
                    'input_centers': simulator.sample_centers(out['input_resized']),
                    # 'target_centers': simulator.sample_centers(out['target_resized'])
                    })

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


##### BOUNDARY PIPELINES
def get_pipeline_boundary_latent(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']

        # Data manipulation
        image, label = batch['image'], batch['contour'] 
        label = dilation3x3(label)

        # Forward pass
        latent = encoder(image)
        reconstruction = decoder(latent)
        if cfg['circular_mask'] is not False:
            reconstruction = reconstruction * cfg['circular_mask']

        # Output dictionary
        out = {'input': image,
               'phosphenes': latent,
               'reconstruction': reconstruction,
               'target': label,
               'target_rgb': label,
               'reconstruction_rgb': reconstruction}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                        func=torch.nn.MSELoss(),
                        arg_names=('reconstruction', 'target'))

    loss_func = CompoundLoss([recon_loss])

    return forward, loss_func

def get_pipeline_unsupervised_boundary(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, label = batch['image'], batch['contour'] 
        label = dilation3x3(label)

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes) # * cfg['circular_mask']

        # Output dictionary
        out = {'input': image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction,
               'target': label,
               'target_rgb': label,
               'reconstruction_rgb': reconstruction}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                        func=torch.nn.MSELoss(),
                        arg_names=('reconstruction', 'target'))

    loss_func = CompoundLoss([recon_loss])

    return forward, loss_func

def get_pipeline_supervised_boundary(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, label = batch['image'], batch['contour'] 
        label = dilation3x3(label)
        unstandardized_image = undo_standardize(image).mean(1, keepdims=True)

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes) # * cfg['circular_mask']

        # Output dictionary
        out = {'input': image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction,
               'target': label,
               'target_resized': resize(label.float() * cfg['circular_mask'], cfg['SPVsize'],),
               'target_rgb': label,
               'reconstruction_rgb': reconstruction,
               'input_resized': resize(unstandardized_image * cfg['circular_mask'], cfg['SPVsize'])}

        # Sample phosphenes and target at the centers of the phosphenes
        out.update({'phosphene_centers': simulator.sample_centers(phosphenes),
                    # 'input_centers': simulator.sample_centers(out['input_resized']),
                    'target_centers': simulator.sample_centers(out['target_resized'])})

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                        func=torch.nn.MSELoss(),
                        arg_names=('reconstruction', 'target'),
                        weight=1-cfg['regularization_weight'])
    
    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_centers', 'target_centers'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

def get_pipeline_boundary_no_decoder(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        simulator = models['simulator']

        # Data manipulation
        image, label = batch['image'], batch['contour'] 
        label = dilation3x3(label)
        unstandardized_image = undo_standardize(image).mean(1, keepdims=True)

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)

        # Output dictionary
        out = {'input': image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'target': label,
               'target_resized': resize(label.float() * cfg['circular_mask'], cfg['SPVsize'],),
               'target_rgb': label,
               'input_resized': resize(unstandardized_image * cfg['circular_mask'], cfg['SPVsize'])}

        # Sample phosphenes and target at the centers of the phosphenes
        out.update({'phosphene_centers': simulator.sample_centers(phosphenes),
                    # 'input_centers': simulator.sample_centers(out['input_resized']),
                    'target_centers': simulator.sample_centers(out['target_resized'])})

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out
    
    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_centers', 'target_centers'))

    loss_func = CompoundLoss([regul_loss])

    return forward, loss_func

def get_pipeline_boundary_supervised_segmentation(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, semantic_map, boundaries = batch['image'] , batch['segmentation_maps'], batch['contour']
        boundaries = dilation3x3(boundaries)
        label_rgb = tensor_to_rgb(semantic_map, cfg['num_classes'])
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
               'target': (semantic_map * cfg['circular_mask']).squeeze(1),
               'target_resized': resize(boundaries.float() * cfg['circular_mask'], cfg['SPVsize'],),
               'target_rgb': label_rgb,
               'reconstruction_rgb': reconstruction_rgb,
               'input_resized': resize(unstandardized_image * cfg['circular_mask'], cfg['SPVsize'])}

        # Sample phosphenes and target at the centers of the phosphenes
        out.update({'phosphene_centers': simulator.sample_centers(phosphenes),
                    # 'input_centers': simulator.sample_centers(out['input_resized']),
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
                          arg_names=('phosphene_centers', 'target_centers'),
                          weight=cfg['regularization_weight'])
    
    loss_func = CompoundLoss([cross_entropy_loss, dice_loss, regul_loss])

    return forward, loss_func

