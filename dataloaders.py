
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
    'brighten', 'color_block', 'color_diffusion', 
    'color_saturation1', 'color_saturation2', 'color_shift', 
    'darken', 'decode_jpeg', 'encode_jpeg', 
    'gaussian_blur', 'high_sharpen', 'impulse_noise', 
    'jitter', 'jpeg', 'jpeg2000', 
    'lens_blur', 'linear_contrast_change', 'mean_shift', 
    'motion_blur', 'multiplicative_noise', 'non_eccentricity_patch', 
    'non_linear_contrast_change', 'pixelate', 'quantization', 
    'white_noise', 'white_noise_cc'
]

distortion_labels = { i:d for i, d in enumerate(dstr,1)}
distortion_labels["OK"]=0

distortion_transforms = { d:getattr(dstr_all, d) for d in dstr}


class VideoFootage(Dataset):
    def __init__(self, image_paths: str, scenario: str = None, display_im: bool = False):

        self.image_paths = image_paths
        self.display_im = display_im

        self.labels = distortion_labels
        self.transforms = distortion_transforms

        self.scenario = scenario
        np.random.seed(seed=1234)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(self.image_paths[idx])
        
        if self.scenario ==None:
            random_label = np.random.choice(list(self.labels.keys()), 1)[0]
        else:
            assert self.scenario in list(self.labels.keys())
            n = self.scenario if idx>self.__len__()//2 else "normal"

        if self.display_im:
            display_image = image
        
        preproc = T.Compose([
            T.CenterCrop(size=min(image.size)),
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image = preproc(image)

        minimize = T.Compose([T.Resize((128,128)),])

        small_image = minimize(image)

        label = self.labels[n]
        if self.display_im:
            if not exists("display_frames"):
                mkdir("display_frames")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(display_image)
            
            plt.savefig(join("display_frames", "frame_{}.jpg".format(idx)), bbox_inches ="tight")
            plt.close()

        return image, small_image, label

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
            # distortion_type = int(im_path.replace(".png", "").split("_")[-2])
            # if distortion_type[0]=="0":
            #     distortion_type = int(distortion_type[1])
            # else:
            #     distortion_type = int(distortion_type)        
            # if quality in [1,2,3]:
            #     label = 0
            # else:
            #     label = 1
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