
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T
import distortions as dstr_all
import matplotlib.pyplot as plt

# dstr = [
#     'gaussian_blur', 'blackout', 'motion_blur', 'brighten', 'color_block', 'color_diffusion', 
#     'color_saturation1', 'color_saturation2', 'color_shift', 
#     'darken', 'decode_jpeg', 'encode_jpeg', 
#     'high_sharpen', 'impulse_noise', 
#     'jitter', 'jpeg', 'jpeg2000', 
#     'lens_blur', 'linear_contrast_change', 'mean_shift', 
#      'multiplicative_noise', 'non_eccentricity_patch', 
#     'non_linear_contrast_change', 'pixelate', 'quantization', 
#     'white_noise', 'white_noise_cc'
# ]



class VideoFootage(Dataset):
    def __init__(self, image_paths: str, 
                 distort: bool = False, 
                 tape: list = [], 
                 window: int = 0, 
                 num_windows: int = 1, 
                 dstr: list=['white_noise'], 
                 dist_sparsity: float = 0.0 ):

        self.dstr = dstr
        self.image_paths = image_paths
        self.transforms = { i:getattr(dstr_all, d) for i,d in enumerate(self.dstr)}
        self.window = window
        self.distort = distort
        if self.distort:
            dist_choices = [i for i in self.transforms.keys()]
            if window==0:
                self.window = self.__len__()-(self.__len__()//2)
                self.tape = [ x for x in range(self.__len__()//2, self.__len__())]
                while len(dist_choices)<self.__len__()-(self.__len__()//2):
                    dist_choices += [i for i in self.transforms.keys()]
                self.dist_choice = dist_choices[:self.__len__()-(self.__len__()//2)]
            else:
                self.num_windows = num_windows
                while len(dist_choices)<window:
                    dist_choices += [i for i in self.transforms.keys()]
                self.dist_choice = dist_choices[:window]
                if tape!=[]:
                    self.tape = tape
                else:
                    all_idx = [ i for i in range(self.__len__())]
                    random_window_start = np.random.choice([ i for i in range(self.__len__()-self.window)], self.num_windows)
                    self.tape = []
                    for win_st in random_window_start:
                        self.tape.extend(all_idx[win_st:win_st+self.window])
            if dist_sparsity!=0.0:
                rm_amount = int(window*dist_sparsity)
                self.tape = np.random.choice([ x for x in self.tape], self.window-rm_amount)
        
            self.dist_map = {id_x:id_dist for id_x,id_dist in zip(self.tape,self.dist_choice)}

    def __len__(self):
        return len(self.image_paths)

    def get_window(self):
        return self.tape

    def __getitem__(self, idx):

        image = Image.open(self.image_paths[idx])
        disp_image = image
        preproc = T.Compose([
            # 720*720
            #T.CenterCrop(size=min(image.size[1:])),

            # The usual image size for ARNIQA -- using min(H,W) is too much !!!
            T.CenterCrop(size=384),
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = preproc(image)

        plt.rcParams["savefig.bbox"] = 'tight'
        plt.rcParams.update({'font.size': 20})

        def disp(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            _, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(20,12))
            legnd = ["original", "lens blur", "motion blur", "blackout"]
            for i, img in enumerate(imgs):
                img = img.detach()
                img = img.permute(1,2,0)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axs[0, i].set_xlabel(legnd[i], fontsize = 30)

        disp_preproc =  T.Compose([T.CenterCrop(size=384), T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        disp_image = disp_preproc(disp_image)
        grid = [disp_image, dstr_all.lens_blur(disp_image), dstr_all.motion_blur(disp_image), dstr_all.darken(disp_image)]
        disp(grid)
        plt.savefig("samples.jpg")
        quit()

        if self.distort and idx in self.tape:
            dist_idx = self.dist_map[idx]
            image = self.transforms[dist_idx](image)
            label = torch.tensor(dist_idx+1)
        else:
            label = torch.tensor(0)

        return image.float(), label

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
            quality = int(im_path.split('_')[2].replace(".png","")[-1])
            label = quality

        big_image = Image.open(im_path)

        preproc = T.Compose([
            T.CenterCrop(size=min(big_image.size)),
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        big_image = preproc(big_image)
        
        label = torch.tensor(label, dtype=torch.long)

        return big_image, label