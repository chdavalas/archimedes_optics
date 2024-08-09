import os
import torch
from sklearn.model_selection import train_test_split
from dataloaders import VideoFootage
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from drift_detector import drift_detector
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
from models_hub import ARNIQA, LSTM_drift
import csv
from copy import deepcopy

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='output.log', 
    filemode='w', 
    format='%(levelname)s:%(message)s',
    level=logging.INFO)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def init_dataloaders(dataset, batch_size=32):

    im_count = len(os.listdir(dataset))
    file_format = "jpg"
    if dataset == "interlaken_inspection":
        file_format = "png"

    image_paths = [
        dataset+'/frame_'+dataset+'_{}.{}'.format(i, file_format) for i in range(im_count)]

    # Split dataset
    dd_paths, test_paths = train_test_split(
        image_paths, test_size=0.95, random_state=42, shuffle=False)

    train_dataset = VideoFootage(dd_paths)
    test_dataset = VideoFootage(test_paths, distort=True, window=100, num_windows=1, dstr=distortion)
    drift_dataset = VideoFootage(dd_paths)
    logger.info(train_dataset.__len__())
    logger.info(test_dataset.__len__())
    logger.info(drift_dataset.__len__())


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    ddetector_loader = DataLoader(
        drift_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, ddetector_loader


def load_arniqa_model(regr_dt: str = "kadid10k"):
    """Load the pre-trained model."""
    # available_datasets = 
    # [
    # "live", "csiq", "tid2013", "kadid10k", 
    # "flive", "spaq", "clive", "koniq10k"
    # ]
    model = ARNIQA(regressor_dataset=regr_dt)

    return model.eval().to(device)

def load_drd(ddetector_dts: DataLoader, dd_type:  str = "mmd"):

    ddetect = drift_detector(detector=dd_type)
    model = ARNIQA().encoder.to(device)
    
    feat_ext = torch.nn.Sequential(deepcopy(model)).eval().to(device)
    
    for param in feat_ext.parameters():
        param.requires_grad = False

    x_ref = []
    for bim, _ in tqdm(ddetector_dts, desc="Drift fit"):
        _, inp = feat_ext(bim.to(device))
        
        inp = inp.argmax(dim=1).unsqueeze(1).float()
        x_ref.append(inp)

    x_ref = torch.cat(x_ref, dim=0)
    ddetect.fit(x_ref)

    return feat_ext, ddetect

def load_lstm_drift(train_dts: DataLoader, num_epochs: int = 5, out_size: int = 2):

    model_type = "lstm"
    model = LSTM_drift(emb_size=128, 
                       hid_size=50, 
                       num_layers=2,
                       class_out_size=out_size).to(device)
    
    if not os.path.exists("drd_{}.pth".format(model_type)):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for _ in tqdm(range(num_epochs), desc="Epoch", position=0):
            model.train()
            running_loss = 0.0
            for bimages, labels in tqdm(train_dts,desc="#b",position=1,leave=False):
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
    help_string = '''usage python3 main.py 
    [pipe_inspection, traffic_inspection, factory_inspection, 
    assembly_line_extreme_inspection, dashcam_inspection, assembly_line_inspection, 
    kadid10k, interlaken_inspection] 
    [batch_size]
    [seed]
    ['gaussian_blur', 'motion_blur', 'brighten', 'color_block', 'color_diffusion', 
    'color_saturation1', 'color_saturation2', 'color_shift', 
    'darken', 'decode_jpeg', 'encode_jpeg', 
    'high_sharpen', 'impulse_noise', 
    'jitter', 'jpeg', 'jpeg2000', 
    'lens_blur', 'linear_contrast_change', 'mean_shift', 
    'multiplicative_noise', 'non_eccentricity_patch', 
    'non_linear_contrast_change', 'pixelate', 'quantization', 
    'white_noise', 'white_noise_cc']'''

    dataset=sys.argv[1]
    global_batch_size = int(sys.argv[2])
    distortion = [sys.argv[3]]
    torch.manual_seed(sys.argv[4])

    assert distortion[0] in [
        'gaussian_blur', 'motion_blur', 'brighten', 'color_block', 'color_diffusion', 
        'color_saturation1', 'color_saturation2', 'color_shift', 
        'darken', 'decode_jpeg', 'encode_jpeg', 
        'high_sharpen', 'impulse_noise', 
        'jitter', 'jpeg', 'jpeg2000', 
        'lens_blur', 'linear_contrast_change', 'mean_shift', 
        'multiplicative_noise', 'non_eccentricity_patch', 
        'non_linear_contrast_change', 'pixelate', 'quantization', 
        'white_noise', 'white_noise_cc'
    ], help_string

    assert dataset in [
        "traffic_inspection",
        "pipe_inspection", 
        "factory_inspection", 
        "assembly_line_extreme_inspection", 
        "assembly_line_inspection", 
        "dashcam_inspection", "interlaken_inspection"] and len(sys.argv)==5, help_string
    
    # INIT DATASETS
    train, test, ddet = init_dataloaders(
        dataset=dataset, batch_size=global_batch_size)

    # LOAD ARNIQA
    model_arniqa = load_arniqa_model().to(device)

    # LOAD ARNIQA+DRIFT DETECTOR
    model_drd, ddetect = load_drd(ddetector_dts=ddet)
     
    # LOAD ARNIQA+ KADID10KLSTM
    model_lstm = load_lstm_drift(train_dts=train)

    model_drd.to(device)

    all_drift_p_values = []
    mean_iqscore_values = []
    all_lstm_mean_values = []

    drift_pred = []
    drift_tar = []

    poor_quality_pred = []
    poor_quality_tar = []

    lstm_drift_pred = []
    lstm_drift_tar = []

    test_dts_with_status_ = tqdm(test, desc="Test", position=0)
    with torch.no_grad():
        im_passed = []; idx=0
        for bimages, labels in test_dts_with_status_:
            classes_, cl_count= torch.unique(labels, sorted=True, return_counts=True)
            logging.info("classes: {}, freq: {}".format(classes_.tolist(), cl_count.tolist()))
            lstm_outputs = model_lstm(bimages.to(device))
            
            arniqa_outputs = compute_quality_score(
                model_arniqa, bimages.to(device),
                )
            _, dd_in = model_drd(bimages.to(device))
            dd_in = dd_in.argmax(dim=1).unsqueeze(1).float()
            pv = ddetect.forward(dd_in)
 
            meaniq = arniqa_outputs.mean().item()
            lstm_labels = torch.where(labels>1, 1, 0)
            lstm_labels = torch.nn.functional.one_hot(lstm_labels.long(), 2)
            lstm_mean = torch.argmax(lstm_outputs, dim=1).float().mean()

            all_drift_p_values.append(pv)
            mean_iqscore_values.append(meaniq)
            all_lstm_mean_values.append(lstm_mean.item())

            logging.info("---------")
            logging.info("drift p-val:{}".format(pv))
            logging.info("mean_iq:{}".format(meaniq))
            logging.info("lstm_mean:{}".format(lstm_mean.item()))
            logging.info("---------")

            # CALCULATE PRED STATS FOR DRIFT
            if pv<0.05:
                drift_pred.append(1)
            else:
                drift_pred.append(0)

            ideal_labels = torch.tensor([0]*bimages.shape[0])

            # CALCULATE PRED STATS FOR LSTM
            if lstm_mean.item()>0.5:
                lstm_drift_pred.append(1)
            else:
                lstm_drift_pred.append(0)

            # CALCULATE PRED STATS FOR IQA
            if meaniq<0.5:
                poor_quality_pred.append(1)
            else:
                poor_quality_pred.append(0)

            # CALCULATE TARGET
            if torch.eq(ideal_labels,labels).sum() < bimages.shape[0]//2:
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
        'dataset':dataset, 
        'distortion_type': distortion[0],
        'method': 'mmd-drift', 
        'precision':precision_score(drift_tar, drift_pred), 
        'recall': recall_score(drift_tar, drift_pred), 
        'f1':f1_score(drift_tar, drift_pred)
    },
    {
        'dataset':dataset, 
        'distortion_type': distortion[0],
        'method': 'arniqa-mean', 
        'precision':precision_score(poor_quality_tar, poor_quality_pred), 
        'recall': recall_score(poor_quality_tar, poor_quality_pred), 
        'f1':f1_score(poor_quality_tar, poor_quality_pred)
    },
        {
        'dataset':dataset, 
        'distortion_type': distortion[0],
        'method': 'lstm-drift', 
        'precision':precision_score(lstm_drift_tar, lstm_drift_pred), 
        'recall': recall_score(lstm_drift_tar, lstm_drift_pred), 
        'f1':f1_score(lstm_drift_tar, lstm_drift_pred)
    },

]

if os.path.exists('stats.csv'):
    with open('stats.csv', 'a', newline='') as csvfile:
        header_name = ['dataset', 'distortion_type', 'method', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=header_name)
        writer.writerows(data)
else:
    with open('stats.csv', 'w', newline='') as csvfile:
        header_name = ['dataset', 'distortion_type', 'method', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=header_name)
        writer.writeheader()
        writer.writerows(data)

