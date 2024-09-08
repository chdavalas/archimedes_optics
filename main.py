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

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='output.log', 
    filemode='w', 
    format='%(levelname)s:%(message)s',
    level=logging.INFO)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

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
            ref_mean = score.min().item() if i==1 else min(ref_mean, score.min().item())

    return model.eval().to(device), ref_mean

def return_feat_ext_output(feat_ext, bim, load_ref):
    if not load_ref:
        _, inp = feat_ext(bim.to(device))
    else:
        inp = feat_ext(bim.to(device))
    out_ = inp.argmax(dim=1).unsqueeze(1).float()
    logger.info(out_.permute(1,0))
    return out_

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

    x_ref = []
    for bim, _ in tqdm(ddetector_dts, desc="Drift fit"):
        inp = return_feat_ext_output(feat_ext, bim, load_ref)
        x_ref.append(inp)

    x_ref = torch.cat(x_ref, dim=0)
    ddetect.fit(x_ref)

    return feat_ext, ddetect

def load_lstm_drift(num_epochs: int = 30, out_size: int = 2):

    model_type = "lstm"
    model = LSTM_drift(emb_size=128, 
                       hid_size=30, 
                       num_layers=2,
                       class_out_size=out_size).to(device)
    
    if not os.path.exists("drd_{}.pth".format(model_type)):

        kadid_paths = glob("kadid10k/images/I*")
        train_d =  kadid10k(kadid_paths)
    
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

        torch.save(model.state_dict(), "drd_{}.pth".format(model_type))

    else:
        logger.info("load from dir")
        model.load_state_dict(
            torch.load("drd_{}.pth".format(model_type))
            )
        model = model.to(device)

    return model.eval()

def compute_quality_score(model, img):
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, return_embedding=False, scale_score=True)
    return score

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
    parser.add_argument('--test-dataset', type=str, help=dataset_list)
    parser.add_argument('--ref-dataset', type=str, help=dataset_list)
    parser.add_argument('--seed', type=int, help='define numpy/pytorch seed for reproducible results', default=42)
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
    
    # INIT DATASETS
    test_ds, test = init_dataloaders(
        dataset_name_split=test_dataset, batch_size=global_batch_size, window_size=window_size, 
        dstr=distortion, distort=True, dist_sparsity=sparsity)
    
    _, rdet = init_dataloaders(
        dataset_name_split=ref_dataset, batch_size=global_batch_size, shuffle=True)

    # _, train = init_dataloaders(
    #     dataset_name_split=ref_dataset, batch_size=global_batch_size, shuffle=True, distort=False, window_size=0, dstr=distortion)

    # LOAD ARNIQA
    model_arniqa, ref_mean = load_arniqa_model(rdet)

    # LOAD ARNIQA+DRIFT DETECTOR
    lf=True
    model_drd, ddetect = load_drd(ddetector_dts=rdet, load_ref=lf, seed=args.seed)
     
    # LOAD ARNIQA+ KADID10K LSTM
    model_lstm = load_lstm_drift()

    all_drift_p_values = []
    mean_iqscore_values = []
    all_lstm_mean_values = []

    drift_pred = []
    drift_tar = []

    poor_quality_pred = []
    poor_quality_tar = []

    lstm_drift_pred = []
    lstm_drift_tar = []

    win_start = 0
    win_count = 0
    flag = True
    test_dts_with_status_ = tqdm(test, desc="Test", position=0)
    with torch.no_grad():
        im_passed = []; idx=0
        for i, (bimages, labels) in enumerate(test_dts_with_status_):
            classes_, cl_count= torch.unique(labels, sorted=True, return_counts=True)
            if classes_.tolist()[-1]!=0:
                status_ = "CORR"
                if flag:
                    win_start = i
                    flag = False
                win_count +=1
            else:
                status_ = "OK"
            logging.info("status: {}, classes: {}, freq: {}".format(status_, classes_.tolist(), cl_count.tolist()))
            lstm_outputs = model_lstm(bimages.to(device))
            
            arniqa_outputs = compute_quality_score(
                model_arniqa, bimages.to(device),
                )
                
            dd_in = return_feat_ext_output(model_drd, bimages, lf)
            pv = ddetect.forward(dd_in)
 
            meaniq = arniqa_outputs.mean().item()
            lstm_mean = torch.argmax(lstm_outputs, dim=1).float().mean()

            all_drift_p_values.append(pv)
            mean_iqscore_values.append(meaniq)
            all_lstm_mean_values.append(lstm_mean.item())

            logging.info("---------")
            logging.info("drift p-val:{}".format(pv))
            logging.info("mean_iq:{}, ref:{}".format(meaniq, ref_mean))
            logging.info("lstm_mean:{}".format(lstm_mean.item()))
            logging.info("Target ISCOR:{}".format(torch.where(labels>0, 1, 0).sum() > labels.shape[0]//2))
            logging.info("---------")

            # CALCULATE PRED STATS FOR DRIFT
            if pv<0.05:
                drift_pred.append(1)
            else:
                drift_pred.append(0)

            # ideal_labels = torch.tensor([0]*bimages.shape[0])

            # CALCULATE PRED STATS FOR LSTM
            if lstm_mean.item()>0.5:
                lstm_drift_pred.append(1)
            else:
                lstm_drift_pred.append(0)

            # CALCULATE PRED STATS FOR IQA
            if meaniq<=ref_mean:
                poor_quality_pred.append(1)
            else:
                poor_quality_pred.append(0)

            # CALCULATE TARGET
            if torch.where(labels>0, 1, 0).sum() > labels.shape[0]//2:
                poor_quality_tar.append(1)
                lstm_drift_tar.append(1)
                drift_tar.append(1)
            else:
                poor_quality_tar.append(0)
                lstm_drift_tar.append(0)
                drift_tar.append(0)

logging.info("--------------------------------------------------")
logging.info("Drift stats")
logging.info("Precision:{}".format(precision_score(drift_tar, drift_pred)))
logging.info("Recall:{}".format(recall_score(drift_tar, drift_pred)))
logging.info("F1:{}".format(f1_score(drift_tar, drift_pred)))
logging.info("--------------------------------------------------")
logging.info("IQA stats")
logging.info("Precision:{}".format(precision_score(poor_quality_tar, poor_quality_pred)))
logging.info("Recall:{}".format(recall_score(poor_quality_tar, poor_quality_pred)))
logging.info("F1:{}".format(f1_score(poor_quality_tar, poor_quality_pred)))
logging.info("--------------------------------------------------")
logging.info("LSTM stats")
logging.info("Precision:{}".format(precision_score(lstm_drift_tar, lstm_drift_pred)))
logging.info("Recall:{}".format(recall_score(lstm_drift_tar, lstm_drift_pred)))
logging.info("F1:{}".format(f1_score(lstm_drift_tar, lstm_drift_pred)))
logging.info("--------------------------------------------------")

import csv

data = [
    {
        'test_dataset':test_dataset, 
        'ref_dataset':ref_dataset, 
        'distortion_type': distortion[0],
        'method': 'mmd-drift',
        'seed': args.seed,
        'window_tape':[win_start, win_start+win_count],
        'window': window_size*(1.0-sparsity),
        'precision':precision_score(drift_tar, drift_pred), 
        'recall': recall_score(drift_tar, drift_pred), 
        'f1':f1_score(drift_tar, drift_pred)
    },
    {
        'test_dataset':test_dataset, 
        'ref_dataset':ref_dataset, 
        'distortion_type': distortion[0],
        'method': 'arniqa-mean', 
        'seed': args.seed,
        'window': window_size*(1.0-sparsity),
        'window_tape':[win_start, win_start+win_count],
        'precision':precision_score(poor_quality_tar, poor_quality_pred), 
        'recall': recall_score(poor_quality_tar, poor_quality_pred), 
        'f1':f1_score(poor_quality_tar, poor_quality_pred)
    },
        {
        'test_dataset':test_dataset, 
        'ref_dataset':ref_dataset, 
        'distortion_type': distortion[0],
        'method': 'lstm-drift', 
        'seed': args.seed,
        'window': window_size*(1.0-sparsity),
        'window_tape':[win_start, win_start+win_count],
        'precision':precision_score(lstm_drift_tar, lstm_drift_pred), 
        'recall': recall_score(lstm_drift_tar, lstm_drift_pred), 
        'f1':f1_score(lstm_drift_tar, lstm_drift_pred)
    },

]

with open('diagnostic_values.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["drift_p_val", "mean_image_quality", "lstm_drift detect", "iqref", "driftref", "lstmref", "window_tape"])
    for dr,iq,ls in zip(all_drift_p_values, mean_iqscore_values, all_lstm_mean_values):
        writer.writerow([dr, iq, ls, ref_mean, 0.05, 0.5, [win_start, win_start+win_count]])
    
if os.path.exists('stats.csv'):
    with open('stats.csv', 'a', newline='') as csvfile:
        header_name = ['test_dataset', 'ref_dataset',  'distortion_type', 'method','seed','window','window_tape', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=header_name)
        writer.writerows(data)
else:
    with open('stats.csv', 'w', newline='') as csvfile:
        header_name = ['test_dataset', 'ref_dataset',  'distortion_type', 'method','seed','window','window_tape', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=header_name)
        writer.writeheader()
        writer.writerows(data)

