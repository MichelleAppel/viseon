import os
from glob import glob

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


def get_lapa_dataset(cfg, split='train'):
    if split == 'train':
        trainset = LaPaDataset(cfg, mode='train')
        valset = LaPaDataset(cfg, mode='val')
        return trainset, valset
    elif split == 'test':
        testset = LaPaDataset(cfg, mode='test')
        return testset
    else:
        raise ValueError(f"Invalid dataset split: {split}. Supported values are 'train', 'val', and 'test'.")

def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    x = torch.arange(h)
    Y, X = torch.meshgrid(x, x, indexing='ij')
    dist_from_center = torch.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

class LaPaDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.directory = cfg['data_directory']
        self.device = cfg['device']
        self.imsize = cfg['imsize']
        self.contour_size = [256, 256]
        self.grayscale = cfg['grayscale']
        self.semantic_labels = 'semantic' in cfg['target']
        self.contour_labels = 'boundary' in cfg['target']
        self.mode = mode
        self.debug_subset = cfg['debug_subset']
        self.retinal_compression = cfg['retinal_compression']
        self.circular_mask = cfg['circular_mask']
        self.fov = cfg['fov']

        # Define paths to images and labels
        self.image_paths, self.label_paths = self._get_paths()

        if self.debug_subset:
            self.image_paths = self.image_paths[:self.debug_subset]
            self.label_paths = self.label_paths[:self.debug_subset]
        
        # Sort to ensure alignment of images and labels
        self.image_paths.sort()
        self.label_paths.sort()

        self.img_transform = self._create_image_transform()
        if self.semantic_labels:
            self.semantic_transform = self._create_semantic_transform()
        if self.contour_labels:
            self.contour_transform = self._create_contour_transform()

        if self.circular_mask:
            self._mask = create_circular_mask(*self.imsize).view(1, *self.imsize)
            self._labelmask = create_circular_mask(*self.contour_size).view(1, *self.contour_size)
        else:
            self._mask = None

    def _get_paths(self):
        if self.mode == 'train':
            image_folder = 'train'
            label_folder = 'train'
        elif self.mode == 'val':
            image_folder = 'val'
            label_folder = 'val'
        elif self.mode == 'test':
            image_folder = 'test'
            label_folder = 'test'
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Supported values are 'train', 'val', and 'test'.")

        image_paths = glob(os.path.join(self.directory, image_folder, 'images', '*.jpg'))
        label_paths = glob(os.path.join(self.directory, label_folder, 'labels', '*.png')) if label_folder else [None] * len(image_paths)

        return image_paths, label_paths

    def _create_image_transform(self):
        if self.retinal_compression:
            from components.RetinalCompression import RetinalCompression
            retinal_compression = RetinalCompression()

            return T.Compose([
                T.Lambda(lambda img: F.center_crop(img, min(img.size))),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
                T.Lambda(lambda img: retinal_compression.single(
                    image=np.array(img),
                    fov=self.fov,
                    out_size=self.imsize[0],
                    inv=0,
                    type=0,
                    show=0,
                    masking=1,
                    series=1,
                    masktype=0
                )),
                T.ToTensor()
            ])
        else:
            return T.Compose([
                T.Lambda(lambda img: F.center_crop(img, min(img.size))),
                T.Resize(self.imsize),
                T.ToTensor()
            ])

    def _create_semantic_transform(self):
        return T.Compose([
            T.Lambda(lambda img: F.center_crop(img, min(img.size))),
            T.Resize(self.contour_size, interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(lambda img: torch.from_numpy(np.array(img)).long())
        ])

    def _create_contour_transform(self):
        contour = lambda im: im.filter(ImageFilter.FIND_EDGES).point(lambda p: p > 1 and 255)
        return T.Compose([
            T.Lambda(lambda img: F.center_crop(img, min(img.size))),
            T.Resize(self.contour_size, interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(contour),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('L') if self.label_paths[idx] is not None else None

        # Apply transformations
        image = self.img_transform(image)
        if self.semantic_labels and label is not None:
            semantic = self.semantic_transform(label)
        else:
            semantic = None

        if self.contour_labels and label is not None:
            contour = self.contour_transform(label)
        else:
            contour = None

        if self._mask is not None:
            image = image * self._mask
            if semantic is not None:
                semantic = semantic * self._mask
            if contour is not None:
                contour = contour * self._labelmask

        # Dictionary containing image, label and contours
        batch = {'image': image.to(self.device)}
        if self.semantic_labels and semantic is not None:
            batch['segmentation_maps'] = semantic.to(self.device)
        if self.contour_labels and contour is not None:
            batch['contour'] = contour.to(self.device)

        return batch
