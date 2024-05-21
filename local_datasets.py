import numpy as np
import os
import pickle
from glob import glob
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.datasets as ds
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageFilter 
import cv2 as cv
import string
import utils
from tqdm import tqdm

def get_ade20k_dataset(cfg):
    trainset = ADE_Dataset(device=cfg['device'],
                                          directory=cfg['data_directory'],
                                          imsize=(128, 128),
                                          load_preprocessed=cfg['load_preprocessed'],
                                          grayscale=cfg['in_channels']==1,
                                          contour_labels=cfg['target']=='boundary',
                                          debug_subset=cfg['debug_subset'],
                                          circular_mask=cfg['circular_mask'])
    valset = ADE_Dataset(device=cfg['device'], directory=cfg['data_directory'],
                                        imsize=(128, 128),
                                        load_preprocessed=cfg['load_preprocessed'],
                                        grayscale=cfg['in_channels']==1,
                                        contour_labels=cfg['target']=='boundary',
                                        validation=True,
                                        debug_subset=cfg['debug_subset'],
                                        circular_mask=cfg['circular_mask'])
    return trainset, valset

def get_bouncing_mnist_dataset(cfg):
    trainset = Bouncing_MNIST(device=cfg['device'],
                                             directory=cfg['data_directory'],
                                             mode=cfg['mode'],
                                             n_frames=cfg['sequence_length'],
                                             imsize=(128, 128))
    valset = Bouncing_MNIST(device=cfg['device'],
                            directory=cfg['data_directory'],
                            mode=cfg['mode'],
                            n_frames=cfg['sequence_length'],
                            imsize=(128, 128),
                            validation=True)
    return trainset, valset


def get_character_dataset(cfg):
    trainset = Character_Dataset(directory=cfg['data_directory'],
                                 device=cfg['device'],
                                 imsize=(128,128),
                                 validation=False,
                                 ver_flip=cfg['flip_vertical'],
                                 hor_flip=cfg['flip_horizontal'])
    
    valset = Character_Dataset(directory=cfg['data_directory'],
                             device=cfg['device'],
                             imsize=(128,128),
                             validation=True,
                             random_pos=False,
                             ver_flip=cfg['flip_vertical'],
                             hor_flip=cfg['flip_horizontal'])
    return trainset, valset

def get_lapa_dataset(cfg):
    trainset = LaPaDataset(directory=cfg['data_directory'], device=cfg['device'], validation=False, circular_mask=cfg['circular_mask'])
    valset = LaPaDataset(directory=cfg['data_directory'], device=cfg['device'], validation=True, circular_mask=cfg['circular_mask'])
    return trainset, valset

def create_circular_mask(h, w, center=None, radius=None, circular_mask=True):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    x = torch.arange(h)
    Y, X = torch.meshgrid(x,x, indexing='ij')
    dist_from_center = torch.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask



class Bouncing_MNIST(Dataset):

    def __init__(self, directory='./datasets/BouncingMNIST',
                 device = torch.device('cuda:0'),
                 mode = 'recon',
                 imsize=(128,128),
                 n_frames=6,
                 validation=False,
                 circular_mask=True):
        super().__init__()
        
        VALIDATION_SPLIT = 0.1 # Fraction of sequences used as validation set

        self.device = device
        self.mode = mode
        self.imsize = imsize
        self.n_frames = n_frames
        full_set = np.load(directory+'mnist_test_seq.npy').transpose(1, 0, 2, 3) # -> (Batch, Frame, Height, Width)
        
        n_val = int(VALIDATION_SPLIT*full_set.shape[0])
        if validation:
            data = torch.from_numpy(full_set[:n_val])
        else:
            data = torch.from_numpy(full_set[n_val:])

        
        _, seq_len, H, W  = data.shape # (original sequence has 20 frames)
        
        # In reconstruction mode, the target is same as input and only one set of images is returned
        if self.mode=='recon':
            
            # Use the remaining frames if the original sequence length (20) fits multiple output sequences (n_frames)
            divisor = seq_len//n_frames 
            full_set = data[:,:n_frames*divisor]
            if divisor>1: 
                data = data.reshape((-1,n_frames,H,W))

        self.data = data.unsqueeze(dim=1) # Add (grayscale) channel 
        
                    
        if circular_mask:
            self._mask = create_circular_mask(*imsize).repeat(1,n_frames,1,1) #(Channel, Frame, Height, Width)
        else:
            self._mask = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.mode == 'recon':
            frames = T.Resize(128)(self.data[i]/255.)
            if self._mask is not None:
                frames = frames*self._mask
            return frames.detach().to(self.device)
        elif self.mode == 'recon_pred':
            input_frames = T.Resize(128)(self.data[i,:,:self.n_frames]/255.)#.to(self.device)
            future_frames = T.Resize(128)(self.data[i,:,self.n_frames:self.n_frames*2]/255.)#.to(self.device)
            
            if self._mask is not None:
                input_frames = input_frames*self._mask
                future_frames = future_frames*self._mask
            return input_frames.detach().to(self.device), future_frames.detach().to(self.device)
            
class ADE_Dataset(Dataset):
    
    def __init__(self, directory='../_Datasets/ADE20K/',
                 device=torch.device('cuda:0'),
                 imsize = (128,128),
                 grayscale = False,
                 normalize = True,
                 contour_labels = False,
                 validation=False,
                 load_preprocessed=False,
                 circular_mask=True,
                 debug_subset=False):
        
        self.validation = validation
        self.contour_labels = contour_labels
        self.normalize = normalize
        self.grayscale = grayscale
        self.device = device
        self.debug_subset = debug_subset
    
        # Image and target tranformations (square crop and resize)
        self.img_transform = T.Compose([T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                        T.Resize(imsize),
                                        T.ToTensor()
                                        ])

        if self.contour_labels:
            # Transformation for target with contour detection
            contour = lambda im: im.filter(ImageFilter.FIND_EDGES).point(lambda p: p > 1 and 255) if self.contour_labels else im
            self.trg_transform = T.Compose([T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                            T.Resize(imsize,interpolation=T.InterpolationMode.NEAREST), # Nearest Neighbour
                                            T.Lambda(contour),
                                            T.ToTensor()
                                            ])
        else:
            # Transformation for target with semantic labels
            self.trg_transform = T.Compose([
                                            T.Lambda(lambda img: F.center_crop(img, min(img.size))),
                                            T.Resize(imsize, interpolation=T.InterpolationMode.NEAREST), # Nearest Neighbour is typically used for segmentation labels to avoid interpolation artifacts
                                            T.Lambda(lambda img: torch.from_numpy(np.array(img)).long()) # For masks
                                        ])

        # Normalize
        self.normalizer = T.Normalize(mean = [0.485, 0.456, 0.406],
                                      std = [0.229, 0.224, 0.225])        
        
        if circular_mask:
            self._mask = create_circular_mask(*imsize).view(1,*imsize)
        else:
            self._mask = None
        
        # RGB converter
        weights=[.3,.59,.11]
        self.to_grayscale = lambda image:torch.sum(torch.stack([weights[c]*image[c,:,:] for c in range(3)],dim=0),
                                                   dim=0,
                                                   keepdim=True)

        self.inputs = []
        self.targets = []
        
        
        if load_preprocessed:
            self.load(directory)
#             print('----Loaded preprocessed data----')
#             print(f'input length: {len(self.inputs)} samples')
        else:
            # Collect files 
            img_files, seg_files = [],[]
            print('----Listing training images----')
            i = 0
            for path, subdirs, files in tqdm(os.walk(os.path.join(directory,'images','ADE','training'))):
            # for path, subdirs, files in os.walk(os.path.join(directory,'training')):
                img_files+= glob(os.path.join(path,'*.jpg'))
                seg_files+= glob(os.path.join(path,'*seg.png'))
                i+=1
                if self.debug_subset and i>=self.debug_subset:
                    break
            val_img_files, val_seg_files, = [],[]
            print('----Listing validation images----')
            i=0
            for path, subdirs, files in tqdm(os.walk(os.path.join(directory,'images','ADE','validation'))):
            # for path, subdirs, files in os.walk(os.path.join(directory,'validation')):
                val_img_files+= glob(os.path.join(path,'*.jpg'))
                val_seg_files+= glob(os.path.join(path,'*seg.png'))
                i+=1
                if self.debug_subset and i>=self.debug_subset:
                    break
            for l in [img_files,seg_files,val_img_files,val_seg_files]:
                l.sort()

            print('Finished listing files')
            # Image and target files
            if validation:
                self.input_files = val_img_files
                self.target_files = val_seg_files
            else:
                self.input_files = img_files
                self.target_files = seg_files

            print('----Preprocessing ADE20K input----')
            for image, target in tqdm(zip(self.input_files, self.target_files),total=len(self.input_files)):
                im = Image.open(image).convert('RGB')
                t = Image.open(target).convert('L')

                # Crop, resize & transform
                x = self.img_transform(im)
                t = self.trg_transform(t)
                                            
                # Additional tranforms:
                if self.normalize:
                    x = self.normalizer(x)
                if self.grayscale:
                    x = self.to_grayscale(x)

                self.inputs += [x]
                self.targets += [t]
            print('----Finished preprocessing----')
            self.save(directory)

    def save(self,directory):
        
        # Make directory if it doesn't exist
        path = os.path.join(directory, 'processed_'+('contour' if self.contour_labels else 'semantic'))
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save files
        mode = '_val' if self.validation else '_train'
        with open(os.path.join(path,f'standardized_processed{mode}_inputs.pkl'),'wb') as f:
            pickle.dump(self.inputs,f)
        with open(os.path.join(path,f'standardized_processed{mode}_targets.pkl'),'wb') as f:
            pickle.dump(self.targets,f)

    def load(self,directory):
        mode = '_val' if self.validation else '_train'
        with open(os.path.join(directory,'processed_'+('contour' if self.contour_labels else 'semantic'),f'standardized_processed{mode}_inputs.pkl'),'rb') as f:
            self.inputs = pickle.load(f)
        with open(os.path.join(directory,'processed_'+('contour' if self.contour_labels else 'semantic'),f'standardized_processed{mode}_targets.pkl'),'rb') as f:
            self.targets = pickle.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        
        x = self.inputs[i]
        t = self.targets[i]

        if self._mask is not None:
            x = x*self._mask
            t = t*self._mask
    
        return x.detach().to(self.device),t.detach().to(self.device)
    
class Character_Dataset(Dataset):
    """ Pytorch dataset containing images of single (synthetic) characters.
    __getitem__ returns an image containing one of 26 ascci lowercase characters, 
    typed in one of 47 fonts(default: 38 train, 9 validation) and the corresponding
    alphabetic index as label.
    """
    def __init__(self,directory = './datasets/Characters/',
                 device=torch.device('cuda:0'),
                 imsize = (128,128),
                 train_val_split = 0.8,
                 validation=False,
                 word_scale=.8,
                 invert = True, 
                 circular_mask=True, 
                 random_pos = True,
                 ver_flip = False,
                 hor_flip = False): 
        
        self.imsize = imsize
        self.tensormaker = T.ToTensor()
        self.device = device
        self.validation = validation
        self.word_scale = word_scale
        self.invert = invert
        self.random_pos = random_pos
        
        if circular_mask:
            self._mask = create_circular_mask(*imsize).view(1,*imsize)
        else:
            self._mask = None
        
        
        
        characters = string.ascii_lowercase
        fonts = glob(os.path.join(directory,'Fonts/*.ttf'))
        
        self.split = round(len(fonts)*train_val_split)
        train_data, val_data = [],[]
        for c in characters:
            for f in fonts[:self.split]:
                train_data.append((f,c))
            for f in fonts[self.split:]:
                val_data.append((f,c))
        self.data = val_data if validation else train_data
        self.classes = characters
        self.lookupletter = {letter: torch.tensor(index) for index, letter in enumerate(characters)}
        self.padding_correction = 6 #By default, PILs ImageDraw function uses excessive padding     
        
        self.ver_flip = ver_flip
        self.hor_flip = hor_flip
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        # Load font and character
        f,c = self.data[i]
        
        # Get label (alphabetic index of character)
        lbl  = self.lookupletter[c]
        
        # Scale character to image
        fontsize = 1
        font = ImageFont.truetype(f,fontsize)
        while max(font.getsize(c))/min(self.imsize) <= self.word_scale:
            fontsize += 1
            font = ImageFont.truetype(f,fontsize)
        fontsize -=1  
        font = ImageFont.truetype(f,fontsize)

        # PIL draw object
        img = Image.fromarray(255*np.ones(self.imsize).astype('uint8'))
        draw = ImageDraw.Draw(img)
        
        if self.random_pos:
            # Calculate left-over space
            textsize = font.getsize(c)
            free_space = np.subtract(self.imsize,textsize)
            free_space += self.padding_correction

            # Draw character at random position
            location = np.random.rand(2)*(free_space)
            location[1]-= self.padding_correction
            draw.text(location,c,(0,),font=font)       
        else:
            location = np.array([*self.imsize])//2
            draw.text(location, c, (0,), font=font, anchor='mm', align='center')

        img = self.tensormaker(img)
        
        
        if self.invert:
            img = 1-img
            
        if self._mask is not None:
            img = img*self._mask
            
        if self.ver_flip:
            img = img.flip([1])
           
        if self.hor_flip:
            img = img.flip([2])

        return img.to(self.device), lbl.to(self.device)


class LaPaDataset(Dataset):
    def __init__(self, directory, 
                device=torch.device('cuda:0'), 
                imsize=(128, 128),
                grayscale=False,
                contour_labels=False, 
                validation=False,
                circular_mask=True,
                debug_subset=False):
        self.directory = directory
        self.device = device
        self.imsize = imsize
        self.grayscale = grayscale
        self.contour_labels = contour_labels
        self.validation = validation
        self.debug_subset = debug_subset

        # Define paths to images and labels
        image_folder = 'train' if not validation else 'val'
        label_folder = 'train' if not validation else 'val'
        
        self.image_paths = glob(os.path.join(directory, image_folder, 'images', '*.jpg'))
        self.label_paths = glob(os.path.join(directory, label_folder, 'labels', '*.png'))

        if debug_subset:
            self.image_paths = self.image_paths[:debug_subset]
            self.label_paths = self.label_paths[:debug_subset]
        
        # Sort to ensure alignment of images and labels
        self.image_paths.sort()
        self.label_paths.sort()

        self.img_transform = T.Compose([T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                T.Resize(imsize),
                                T.ToTensor()
                                ])

        if self.contour_labels:
            # Transformation for target with contour detection
            contour = lambda im: im.filter(ImageFilter.FIND_EDGES).point(lambda p: p > 1 and 255) if self.contour_labels else im
            self.trg_transform = T.Compose([T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                            T.Resize(imsize,interpolation=T.InterpolationMode.NEAREST), # Nearest Neighbour
                                            T.Lambda(contour),
                                            T.ToTensor()
                                            ])
        else:
            # Transformation for target with semantic labels
            self.trg_transform = T.Compose([
                                            T.Lambda(lambda img: F.center_crop(img, min(img.size))),
                                            T.Resize(imsize, interpolation=T.InterpolationMode.NEAREST), # Nearest Neighbour is typically used for segmentation labels to avoid interpolation artifacts
                                            T.Lambda(lambda img: torch.from_numpy(np.array(img)).long()) # For masks
                                        ])


        if circular_mask:
            self._mask = create_circular_mask(*imsize).view(1, *imsize)
        else:
            self._mask = None


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('L')  # Assuming labels are grayscale

        # Apply transformations
        image = self.img_transform(image)
        label = self.trg_transform(label)

        if self._mask:
            image = image * self._mask
            label = label * self._mask

        return image.to(self.device), label.to(self.device)
