
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os.path import exists, join
from os import mkdir
import torchvision.transforms.v2 as T

def salt_n_pepper_noise(image):
    (w, h) = image.size
    pixel_range = np.random.randint((w*h)//100, (w*h)//2)
    val = np.random.choice([0,1], 1)[0]
    for i in range(min(w*h-1, pixel_range)): 
        y_coord=np.random.randint(0, w - 1) 
        x_coord=np.random.randint(0, h - 1)
        image.putpixel((y_coord,x_coord), (val, val, val))

    return image

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
    
class DroneFootage(Dataset):
    def __init__(self, image_paths, scenario=None, display_im=False):

        self.image_paths = image_paths
        self.display_im = display_im
        self.blur = T.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11))
        self.elastic_noise = T.ElasticTransform(alpha=235)

        self.labels = {
            "normal":torch.tensor(0, dtype=torch.long), 
            "blur":torch.tensor(1, dtype=torch.long),
            "noise":torch.tensor(2, dtype=torch.long),
            "SnP":torch.tensor(3, dtype=torch.long)
            }
        
        self.transforms = {
            "normal": (lambda x :x), 
            "blur": ( lambda x :self.blur(x) ),
            "noise": ( lambda x : self.elastic_noise(x)),
            "SnP":( lambda x : salt_n_pepper_noise(x))
            }
        self.scenario = scenario
        np.random.seed(seed=1234)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(self.image_paths[idx])
        
        if self.scenario ==None:
            n = np.random.choice(list(self.labels.keys()), 1)[0]
        else:
            assert self.scenario in list(self.labels.keys())
            n = self.scenario if idx>self.__len__()//2 else "normal"

        bright_rand = T.Compose([T.ColorJitter(brightness=(0.7, 2.2))])
        image = bright_rand(self.transforms[n](image))
        if self.display_im:
            display_image = image
        
        preproc = T.Compose([
            T.CenterCrop(size=min(image.size)),
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        # EXIF DATA ROTATES THE ORIGINAL IMAGE
        # large_image = T.functional.rotate(preproc(image), angle=90) 
        
        image = preproc(image)

        ds_preproc = T.Compose([
            T.Resize((128,128)),
        ])
        small_image = ds_preproc(image)

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

