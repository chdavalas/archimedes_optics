
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os.path import exists, join
from os import mkdir
import torchvision.transforms.v2 as T

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
            # distortion_type = im_path.replace(".png", "").split("_")[-2]
            # if distortion_type[0]=="0":
            #     distortion_type = int(distortion_type[1])
            # else:
            #     distortion_type = int(distortion_type)

        
            if quality in [1,2,3]:
                label = 1
            else:
                label = 2

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