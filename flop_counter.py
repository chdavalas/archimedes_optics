from calflops import calculate_flops, calculate_flops_hf
from torchvision import models
import argparse
import os
import torch
from dataloaders import VideoFootage, kadid10k
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from drift_detector import drift_detector
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from models_hub import ARNIQA, LSTM_drift
import csv
from copy import deepcopy
import numpy as np
import models_hub
from glob import glob
from models_hub import ARNIQA

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='output.log', 
    filemode='w', 
    format='%(levelname)s:%(message)s',
    level=logging.INFO)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# device =  "cpu"

def init_dataloaders(dataset_name_split, batch_size=32, window_size=100, 
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

    return _dataset, _loader

def load_arniqa_model(ddetector_dts: DataLoader, regr_dt: str = "kadid10k"):
    """Load the pre-trained model."""
    # available_datasets = 
    # [
    # "live", "csiq", "tid2013", "kadid10k", 
    # "flive", "spaq", "clive", "koniq10k"
    # ]
    model = ARNIQA(regressor_dataset=regr_dt).to(device)
    
    for i, (bimage, _) in enumerate(tqdm(ddetector_dts, desc="Calc ref image quality"),1):
        with torch.no_grad(), torch.cuda.amp.autocast():
            score = model(bimage.to(device), return_embedding=False, scale_score=True)
            ref_iq = score.min().item() if i==1 else min(ref_iq, score.min().item())

    return model.eval().to(device), ref_iq

def return_feat_ext_output(feat_ext, bim, load_ref):
    if not load_ref:
        _, inp = feat_ext(bim.to(device))
    else:
        inp = feat_ext(bim.to(device))
    out_ = inp.argmax(dim=1).unsqueeze(1).float()
    logger.info(out_.permute(1,0))
    return out_

def return_class_mean(feat_ext, bim):
    inp = feat_ext(bim.to(device))
    inp = inp.argmax(dim=1)
    inp = torch.where(inp>0,1.,0.)
    inp_sum = inp.sum()
    return inp_sum/inp.shape[0]

def load_drd(ddetector_dts: DataLoader, dd_type:  str = "mmd", load_ref=False, seed=42):

    ddetect = drift_detector(detector=dd_type)
    
    if not load_ref or not os.path.exists("ref_model_seed_{}.pth".format(seed)):
        logger.info("DRIFT DETECT: ref model NOT FOUND/NOT REQUESTED")
        model = ARNIQA().encoder.to(device)
        feat_ext = torch.nn.Sequential(deepcopy(model)).eval().to(device)
    else:
        logger.info("DRIFT DETECT: ref model LOADED")
        model = models_hub.ResNet18(head_dim=10).to(device)
        model.load_state_dict(torch.load("ref_model_seed_{}.pth".format(seed)))
        feat_ext = torch.nn.Sequential(deepcopy(model)).eval().to(device)

    
    for param in feat_ext.parameters():
        param.requires_grad = False

    # x_ref = []
    # for bim, _ in tqdm(ddetector_dts, desc="Drift fit"):
    #     inp = return_feat_ext_output(feat_ext, bim, load_ref)
    #     x_ref.append(inp)

    # x_ref = torch.cat(x_ref, dim=0)
    # ddetect.fit(x_ref)

    return feat_ext, ddetect

def load_lstm_drift(num_epochs: int = 30, out_size: int = 2, seed: int = 44):

    model_type = "lstm"
    model = LSTM_drift(emb_size=128, 
                       hid_size=50, 
                       num_layers=2,
                       class_out_size=out_size
                       ).to(device)
    
    if not os.path.exists("ref_lstm_seed_{}.pth".format(seed)):
        kadid_paths = glob("kadid10k/images/I*_*")
        is_good_im = (lambda x: int(x.split('_')[2].replace(".png","")[-1])==1)
        is_bad_im = (lambda x: int(x.split('_')[2].replace(".png","")[-1])==5)

        #kadid_paths = [pth for pth in kadid_paths if is_good_im(pth) or is_bad_im(pth)]
        kadid_paths = [pth for pth in kadid_paths if is_good_im(pth) or is_bad_im(pth) ]
        train_d = kadid10k(kadid_paths)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for _ in tqdm(range(num_epochs), desc="LSTM Epoch", position=0):
            model.train()
            running_loss = 0.0
            train_dts = DataLoader(train_d, batch_size=32, shuffle=True)

            for bimages, labels in tqdm(train_dts,desc="#b",position=1,leave=False):
                # Accepting quality 0,1 as good !!!
                labels = torch.where(labels>1, 1, 0)

                outputs = model(bimages.to(device))

                labels = torch.nn.functional.one_hot(labels.long(), out_size)
                loss = criterion(outputs.to(device), labels.float().to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            logger.info(f"Loss: {running_loss/len(train_dts):.4f}")

        torch.save(model.state_dict(),"ref_lstm_seed_{}.pth".format(seed))

    else:
        logger.info("load from dir")
        model.load_state_dict(
            torch.load("ref_lstm_seed_{}.pth".format(seed))
            )
        model = model.to(device)

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
    'jitter', 'jpeg', 'jpeg2000', 'blackout',
    'lens_blur', 'linear_contrast_change', 'mean_shift', 
    'multiplicative_noise', 'non_eccentricity_patch', 
    'non_linear_contrast_change', 'pixelate', 'quantization', 
    'white_noise', 'white_noise_cc. You can also choose multiple distortions which will be applied randomly within a window'''


    parser = argparse.ArgumentParser(description='testing camera diagnostics')
    parser.add_argument('--test-dataset', default="dud", type=str, help=dataset_list)
    parser.add_argument('--ref-dataset', default="dud", type=str, help=dataset_list)
    parser.add_argument('--seed', type=int, help='define numpy/pytorch seed for reproducible results', default=44)
    parser.add_argument('--batch-size', type=int, help='define batch size for training/referencing/testing', default=24)
    parser.add_argument('--window', type=int, help='corruption window, if >0 the window is the last half of the testing dataset', default=0)
    parser.add_argument('--window-sparsity', type=float, help='amount of distortion within an window in the form of percent [0.0, 1.0]', default=0.0)
    parser.add_argument('--distortion-type', type=str, help=distortion_list, default="white_noise")
    args = parser.parse_args()
    
    test_dataset = args.test_dataset.split(',')
    if len(test_dataset)==2: test_dataset.insert(1, '0')
    ref_dataset = args.ref_dataset.split(',')
    if len(ref_dataset)==2: ref_dataset.insert(1, '0')

    global_batch_size = args.batch_size
    window_size = args.window
    sparsity = args.window_sparsity
    distortion = args.distortion_type.split(',')

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)


    test_ds, test = init_dataloaders(
    dataset_name_split=test_dataset, batch_size=global_batch_size, window_size=window_size, 
    dstr=distortion, distort=True, dist_sparsity=sparsity)

    _, rdet = init_dataloaders(
    dataset_name_split=ref_dataset, batch_size=global_batch_size, shuffle=True)



    # LOAD ARNIQA
    model_arniqa, ref_iq = load_arniqa_model(rdet)

    # LOAD DRIFT DETECTOR
    lf=True
    model_drd, ddetect = load_drd(ddetector_dts=rdet, load_ref=lf, seed=args.seed)

    # LOAD ARNIQA + KADID10K LSTM
    model_lstm = load_lstm_drift(seed=args.seed)
    full_arniqa_flops = 0
    full_arniqa_macs = 0
    full_arniqa_params = 0

    batch_size = 24
    input_shape = (batch_size, 3, 384, 384)
    
    flops, macs, params = calculate_flops(model=model_arniqa.encoder, input_shape=input_shape, output_as_string=True, output_precision=4, print_results=False)
    print("ARNIQA large FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    full_arniqa_flops+=float(flops.replace(" GFLOPS", ""))
    full_arniqa_macs+=float(macs.replace(" GMACs", ""))
    full_arniqa_params+=float(params[:-1])

    small_shape = (batch_size, 3, 128, 128)
    flops, macs, params = calculate_flops(model=model_arniqa.encoder, input_shape=small_shape, output_as_string=True, output_precision=4, print_results=False)
    print("ARNIQA small FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    full_arniqa_flops+=float(flops.replace(" GFLOPS", ""))
    full_arniqa_macs+=float(macs.replace(" GMACs", ""))
    full_arniqa_params+=float(params[:-1])

    regressor_lookalike = nn.Linear(in_features=4096, out_features=1, bias=True).to(device)
    regressor_shape = (batch_size, 4096)
    flops, macs, params = calculate_flops(model=regressor_lookalike, input_shape=regressor_shape, output_as_string=True, output_precision=4, print_results=False, output_unit="G")
    print("ARNIQA regr FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    full_arniqa_flops+=float(flops.replace(" GFLOPS", ""))
    full_arniqa_macs+=float(macs.replace(" GMACs", ""))
    full_arniqa_params+=float(params[:-1])
    print("-----------------------------------------------------------")
    print("ARNIQA TOTAL FLOPs:%s   MACs:%s   Params:%s" %(round(full_arniqa_flops,3), round(full_arniqa_macs,3), round(full_arniqa_params,3)))
    print("NVIDIA Jetson nano FP32 speed:%s" %(round(full_arniqa_flops,3)/235.8))

    input_shape = (batch_size, 3, 384, 384)
    flops, macs, params = calculate_flops(model=model_drd, input_shape=input_shape, output_as_string=True, output_precision=4, print_results=False)
    print("DRD FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    print("NVIDIA Jetson nano FP32 speed:%s" %(round(flops,3)/235.8))

    input_shape = (batch_size, 3, 384, 384)
    flops, macs, params = calculate_flops(model=model_lstm, input_shape=input_shape, output_as_string=True, output_precision=4, print_results=False)
    print("LSTM FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    print("NVIDIA Jetson nano FP32 speed:%s" %(round(flops,3)/235.8))
    print("-----------------------------------------------------------")


    # import time
    # import statistics
    # device = "cpu"
    # bim = torch.randn(batch_size, 3, 384, 384).to(device)
    # model_arniqa = ARNIQA(regressor_dataset="kadid10k").to(device)
    # ##########################################################################################################
    # for _ in range(5):
    #     _ = model_arniqa(bim)
    # num_iterations = 5
    # inference_times = []
    # for _ in range(num_iterations):
    #     start_time = time.time()
    #     output = model_arniqa(bim)
    #     end_time = time.time()
    #     inference_times.append(end_time - start_time)
    # average_inference_time = statistics.mean(inference_times)
    # print(f"Average inference time for ARNIQA full: {average_inference_time:.4f} seconds")
    # ##########################################################################################################
    # for _ in range(5):
    #     _ = model_drd.to(device)(bim)
    # num_iterations = 5
    # inference_times = []
    # for _ in range(num_iterations):
    #     start_time = time.time()
    #     output = model_drd.to(device)(bim)
    #     end_time = time.time()
    #     inference_times.append(end_time - start_time)
    # average_inference_time = statistics.mean(inference_times)
    # print(f"Average inference time for Class mean / Drift detect: {average_inference_time:.4f} seconds")
    # ##########################################################################################################
    # for _ in range(5):
    #     _ = model_lstm.to(device)(bim)
    # num_iterations = 5
    # inference_times = []
    # for _ in range(num_iterations):
    #     start_time = time.time()
    #     output = model_lstm.to(device)(bim)
    #     end_time = time.time()
    #     inference_times.append(end_time - start_time)
    # average_inference_time = statistics.mean(inference_times)
    # print(f"Average inference time for LSTM (big image): {average_inference_time:.4f} seconds")