import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class DatasetISIC(Dataset):
    '''ISIC images/masks dataset'''
    def __init__(self, folder, transform=None):
        '''
        Args:
            folder (string): Directory with images and masks folders
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        images_folder = folder + '/images'
        masks_folder = folder + '/masks'
        
        images_filenames = sorted(os.listdir(images_folder))
        masks_filenames = sorted(os.listdir(masks_folder))
        
        self.images_paths = [os.path.join(images_folder, name) for name in images_filenames]
        self.masks_paths = [os.path.join(masks_folder, name) for name in masks_filenames]
        
        self.transform = transform
    
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = Image.open(self.images_paths[index])
        mask = Image.open(self.masks_paths[index])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        sample = {'image': image, 'mask': mask}
        
        return sample