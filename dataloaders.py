
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T
import distortions as dstr_all
import sys

# dstr = [
#     'gaussian_blur', 'motion_blur', 'brighten', 'color_block', 'color_diffusion', 
#     'color_saturation1', 'color_saturation2', 'color_shift', 
#     'darken', 'decode_jpeg', 'encode_jpeg', 
#     'high_sharpen', 'impulse_noise', 
#     'jitter', 'jpeg', 'jpeg2000', 
#     'lens_blur', 'linear_contrast_change', 'mean_shift', 
#      'multiplicative_noise', 'non_eccentricity_patch', 
#     'non_linear_contrast_change', 'pixelate', 'quantization', 
#     'white_noise', 'white_noise_cc'
# ]



np.random.seed(seed=int(sys.argv[3]))

class VideoFootage(Dataset):
    def __init__(self, image_paths: str, 
                 distort: bool = False, 
                 tape: list = [], 
                 window: int = 50, 
                 num_windows: int = 3, 
                 dstr: list=['multiplicative_noise']):

        self.dstr = dstr
        self.image_paths = image_paths
        self.transforms = { i:getattr(dstr_all, d) for i,d in enumerate(self.dstr)}
        self.window = window
        self.distort = distort
        if self.distort:
            self.num_windows = num_windows
            dist_choices = np.random.choice([i for i in self.transforms.keys()], num_windows)
            self.dist_choice = sorted(np.repeat(dist_choices, window))
            if tape!=[]:
                self.tape = tape
            else:
                all_idx = [ i for i in range(self.__len__())]
                random_window_start = np.random.choice([ i for i in range(self.__len__()-self.window)], self.num_windows)
                self.tape = []
                for win_st in random_window_start:
                    self.tape.extend(all_idx[win_st:win_st+self.window])

        
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
        if self.distort and idx in self.tape:
            dist_idx = self.dist_choice.pop()
            image = self.transforms[dist_idx](image)
            # plt.imshow(image.permute(1,2,0))
            # plt.show()
            label = torch.tensor(dist_idx+1)
        else:
            label = torch.tensor(0)

        # image = T.RandomRotation([-10,10])(image)
        # image = T.ColorJitter(brightness=(0.7, 1.4))(image)
        return image.float(), label

# class kadid10k(Dataset):
#     def __init__(self, image_paths: str):
#         self.image_paths = image_paths

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         im_path = self.image_paths[idx]

#         if len(im_path.replace(".png", "").split("/")[-1])==3:
#             label = 1
#         else:
#             quality = int(im_path.split('.')[0][-1])
#             label = quality

#         big_image = Image.open(im_path)

#         preproc = T.Compose([
#             T.CenterCrop(size=min(big_image.size)),
#             T.ToImage(), T.ToDtype(torch.float32, scale=True),
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#         big_image = preproc(big_image)
        
#         label = torch.tensor(label, dtype=torch.long)

#         return big_image, label