
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os.path import exists, join
from os import mkdir
import torchvision.transforms.v2 as T
import distortions as dstr_all

dstr = [
    'gaussian_blur', 'motion_blur', 'brighten', 'color_block', 'color_diffusion', 
    'color_saturation1', 'color_saturation2', 'color_shift', 
    'darken', 'decode_jpeg', 'encode_jpeg', 
    'high_sharpen', 'impulse_noise', 
    'jitter', 'jpeg', 'jpeg2000', 
    'lens_blur', 'linear_contrast_change', 'mean_shift', 
     'multiplicative_noise', 'non_eccentricity_patch', 
    'non_linear_contrast_change', 'pixelate', 'quantization', 
    'white_noise', 'white_noise_cc'
]

# distortion_labels = { i:d for i, d in enumerate(dstr,1)}
# distortion_labels[0]=[0]
np.random.seed(seed=1234)

distortion_transforms = { i:getattr(dstr_all, d) for i,d in enumerate(dstr)}

class VideoFootage(Dataset):
    def __init__(self, image_paths: str, distort: str = ""):

        self.image_paths = image_paths
        # self.display_im = display_im

        # self.labels = distortion_labels
        self.transforms = distortion_transforms
        self.distort = distort


        self.last_half = self.__len__()//2
        self.end = self.__len__()
        self.random_dist = np.random.choice([i for i in range(self.last_half, self.end)], self.last_half)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(self.image_paths[idx])

        preproc = T.Compose([
            T.CenterCrop(size=min(image.size[1:])),
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = preproc(image)
        # if self.distort and idx in self.random_dist:
        if idx > self.last_half and self.distort=="last":
            image = self.transforms[0](image)
            label = torch.tensor(2)
        elif self.distort=="rand":
            if np.random.choice([0,1,2],1)[0]==0:
                image = self.transforms[1](image)
                label = torch.tensor(0)
            if np.random.choice([0,1,2],1)[0]==1:
                image = self.transforms[0](image)
                label = torch.tensor(2)
            else:
                label = torch.tensor(1)
        else:
            label = torch.tensor(1)

        # image = T.RandomRotation([-10,10])(image)

        return image, label

class kadid10k(Dataset):
    def __init__(self, image_paths: str):
        self.image_paths = image_paths


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        im_path = self.image_paths[idx]

        if len(im_path.replace(".png", "").split("/")[-1])==3:
            label = 1
        else:
            quality = int(im_path.split('.')[0][-1])
            label = quality

        big_image = Image.open(im_path)

        preproc = T.Compose([
            T.CenterCrop(size=min(big_image.size)),
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        ds_preproc = T.Compose([
            T.Resize((128,128)),
        ])

        big_image = preproc(big_image)
        small_image = ds_preproc(big_image)
        
        label = torch.tensor(label, dtype=torch.long)

        return big_image, small_image, label