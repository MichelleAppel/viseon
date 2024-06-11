import numpy as np
import os
from glob import glob
import torch

from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image, ImageFilter 


def get_lapa_dataset(cfg):
    trainset = LaPaDataset(cfg, validation=False)
    valset = LaPaDataset(cfg, validation=True)
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

class LaPaDataset(Dataset):
    def __init__(self, cfg, validation=False):
        self.directory = cfg['data_directory']
        self.device = cfg['device']
        self.imsize = cfg['imsize']
        self.grayscale = cfg['grayscale']
        self.semantic_labels = 'semantic' in cfg['target']
        self.contour_labels = 'boundary' in cfg['target']
        self.validation = validation
        self.debug_subset = cfg['debug_subset']
        self.retinal_compression = cfg['retinal_compression']
        self.circular_mask = cfg['circular_mask']
        self.fov = cfg['fov']

        # Define paths to images and labels
        image_folder = 'train' if not validation else 'val'
        label_folder = 'train' if not validation else 'val'
        
        self.image_paths = glob(os.path.join(self.directory, image_folder, 'images', '*.jpg'))
        self.label_paths = glob(os.path.join(self.directory, label_folder, 'labels', '*.png'))

        if self.debug_subset:
            self.image_paths = self.image_paths[:self.debug_subset]
            self.label_paths = self.label_paths[:self.debug_subset]
        
        # Sort to ensure alignment of images and labels
        self.image_paths.sort()
        self.label_paths.sort()

        if self.retinal_compression:
            from components.RetinalCompression import RetinalCompression
            retinal_compression = RetinalCompression()

            self.img_transform = T.Compose([
                                        T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                        T.Resize((256, 256)),
                                        T.Lambda(lambda img: retinal_compression.single(image=np.array(img), 
                                                                                 fov=self.fov, 
                                                                                 out_size=self.imsize[0], 
                                                                                 inv=0, 
                                                                                 type=0, 
                                                                                 show=0,
                                                                                 masking=1, 
                                                                                 series=1, 
                                                                                 masktype=0)),
                                        T.ToTensor()
                                    ])
        else:
            
            self.img_transform = T.Compose([T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                    T.Resize(self.imsize),
                                    T.ToTensor()
                                    ])

        # Transformation for target with semantic labels
        if self.semantic_labels:
            self.semantic_transform = T.Compose([
                                            T.Lambda(lambda img: F.center_crop(img, min(img.size))),
                                            T.Resize(self.imsize, interpolation=T.InterpolationMode.NEAREST), # Nearest Neighbour is typically used for segmentation labels to avoid interpolation artifacts
                                            T.Lambda(lambda img: torch.from_numpy(np.array(img)).long()) # For masks
                                        ])

        if self.contour_labels:
            # Transformation for target with contour detection
            contour = lambda im: im.filter(ImageFilter.FIND_EDGES).point(lambda p: p > 1 and 255) if self.contour_labels else im
            self.contour_transform = T.Compose([
                                            T.Lambda(lambda img:F.center_crop(img, min(img.size))),
                                            T.Resize(self.imsize,interpolation=T.InterpolationMode.NEAREST),
                                            T.Lambda(contour),
                                            T.ToTensor()
                                        ])

        if self.circular_mask:
            self._mask = create_circular_mask(*self.imsize).view(1, *self.imsize)
        else:
            self._mask = None


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('L') 

        # Apply transformations
        image = self.img_transform(image)
        if self.semantic_labels:
            semantic = self.semantic_transform(label)
        else:
            semantic = None

        if self.contour_labels:
            contour = self.contour_transform(label)
        else:    
            contour = None

        if self._mask is not None:
            image = image * self._mask
            if semantic is not None:
                semantic = semantic * self._mask
            if contour is not None:
                contour = contour * self._mask

        # Dictionary containing image, label and contours
        batch = {'image': image.to(self.device)} 
        if self.semantic_labels:
            batch['segmentation_maps'] = semantic.to(self.device)
        if self.contour_labels:
            batch['contour'] = contour.to(self.device)

        return batch