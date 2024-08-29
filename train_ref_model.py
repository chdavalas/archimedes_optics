import os
import torch
from dataloaders import VideoFootage
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import logging
from torchvision.models import resnet18
import argparse
import numpy as np
import models_hub

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='output.log', 
    filemode='w', 
    format='%(levelname)s:%(message)s',
    level=logging.INFO)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def init_train_dataloaders(dataset_name_split, batch_size=64, window_size=0, 
                     shuffle=False, drop_last=True, distort=False, 
                     num_windows=1, dist_sparsity=0.0, dstr=["white_noise"]):

    im_count = len(os.listdir(dataset_name_split[0]))

    if len(dataset_name_split)==3:
        dataset, start, stop = dataset_name_split
        start, stop = int(start), int(stop)
        if stop==0: stop=im_count

    else:
        dataset, start, stop = dataset_name_split[0], 0, im_count
        
    file_format = "jpg"
    if dataset in ["interlaken_inspection", "zurich_inspection"]:
        file_format = "png"

    image_paths = [
        dataset+'/frame_'+dataset+'_{}.{}'.format(i, file_format) for i in range(im_count)]
    
    image_paths = image_paths[start:stop]

    _dataset = VideoFootage(
        image_paths, distort=distort, window=window_size, 
        num_windows=num_windows, dist_sparsity=dist_sparsity, dstr=dstr)
    
    logger.info("name:{}, total frames:{} from {} to {}".format(dataset, _dataset.__len__(), start, stop))

    _loader = DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return _loader


def create_ref_model(dataset_name_split, distortions, dsp, num_epochs=50, dim=10, window=100, seed=42):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    model = models_hub.ResNet18(head_dim=dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    if not os.path.exists("ref_model_seed_{}.pth".format(seed)):
        for _ in tqdm(range(num_epochs), desc="Epoch", position=0):
            model.train()
            running_loss = 0.0
            train_dts = init_train_dataloaders(dataset_name_split=dataset_name_split, 
                                            dstr=distortions, 
                                            dist_sparsity=dsp, 
                                            shuffle=True, distort=True, window_size=window)

            for images, labels in tqdm(train_dts, desc="batch", position=1, leave=False):
                optimizer.zero_grad()
                outputs = model(images.to(device))
                loss = criterion(outputs, labels.to(device).long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            logger.info(f"Loss: {running_loss/len(train_dts):.4f}")

            torch.save(model.state_dict(), "ref_model_seed_{}.pth".format(seed))
    else:
        logger.info("REF model exists!!!")

    return model.eval()


if __name__ == "__main__":
    dataset_list = '''Choose between: pipe_inspection, traffic_inspection, factory_inspection, 
    assembly_line_extreme_inspection, dashcam_inspection, assembly_line_inspection, 
    kadid10k, interlaken_inspection, zurich_inspection. You can also define a split (optionally) with comma separated values (format: dataset,start,stop OR dataset,stop) 
    i.e traffic_inspection,100,200 
    OR uav_inspection,100 (same as uav_inspection,0,100) 
    OR dashcam_inspection,200,0 (same as dashcam_inspetion,200,len(dashcam_inspection))'''
    
    distortion_list=''' choose between:
    'gaussian_blur', 'motion_blur', 'brighten', 'color_block', 'color_diffusion', 
    'color_saturation1', 'color_saturation2', 'color_shift', 
    'darken', 'decode_jpeg', 'encode_jpeg', 
    'high_sharpen', 'impulse_noise', 
    'jitter', 'jpeg', 'jpeg2000', 
    'lens_blur', 'linear_contrast_change', 'mean_shift', 
    'multiplicative_noise', 'non_eccentricity_patch', 
    'non_linear_contrast_change', 'pixelate', 'quantization', 
    'white_noise', 'white_noise_cc. You can also choose multiple distortions which will be applied randomly within a window'''


    parser = argparse.ArgumentParser(description='testing camera diagnostics')
    parser.add_argument('--ref-dataset', type=str, help=dataset_list)
    parser.add_argument('--seed', type=int, help='define numpy/pytorch seed for reproducible results', default=42)
    parser.add_argument('--num-epochs', type=int, help='train epochs for reference model', default=100)
    parser.add_argument('--distortion-type', type=str, help=distortion_list, default="white_noise,lens_blur,gaussian_blur")
    parser.add_argument('--window-sparsity', type=float, help='amount of distortion within an window in the form of percent [0.0, 1.0]', default=0.15)
    parser.add_argument('--window', type=int, help='distortion window', default=200)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    ref_dataset = args.ref_dataset.split(',')
    if len(ref_dataset)==2: ref_dataset.insert(1, '0')

    distortion = args.distortion_type.split(',')
    create_ref_model(ref_dataset, num_epochs=args.num_epochs, distortions=distortion, dsp=args.window_sparsity, window=args.window, seed=args.seed)