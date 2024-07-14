
import urllib.request
import os
import torch
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataloaders import kadid10k, VideoFootage
from losses import nt_xent_loss
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from drift_detector import drift_detector
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

from models_hub import ResNet, ARNIQA
import csv
import matplotlib.pyplot as plt
from dotmap import DotMap
from copy import deepcopy

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='output.log', 
    filemode='w', 
    format='%(levelname)s:%(message)s',
    level=logging.INFO)

torch.manual_seed(0)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def init_dataloaders(dataset="kadid10k", scenario="SnP", batch_size=32):

    # Create datasets and loaders
    if dataset == "kadid10k":
        url = "https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip"
        file_name = "kadid10k.zip"

        if not os.path.exists("kadid10k"):
            logger.info("Downloading data ...")
            urllib.request.urlretrieve(url, file_name)
            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                logger.info("Extracting data ...")
                zip_ref.extractall('.')
        

        alphanumeric  = ["0{}".format(i) for i in range(1,10)]
        alphanumeric += ["{}".format(i) for i in range(10,82)]

        # Example image paths and labels
        image_paths = [
            'kadid10k/images/I'+i+'_'+j+'_'+k+'.png'
            for k in [alphanumeric[0]]+[alphanumeric[4]]
            for i in alphanumeric
            for j in alphanumeric[:25]
        ]


        pristine_images = ['kadid10k/images/I'+i+'.png' for i in alphanumeric]

        # Split dataset
        train_paths, test_paths = train_test_split(
            image_paths, test_size=0.4, random_state=42, shuffle=True)

        prist_train_paths, prist_test_paths = train_test_split(
            pristine_images, test_size=0.4, random_state=42, shuffle=True)

        train_paths += prist_train_paths
        test_paths = prist_test_paths + sorted(test_paths, key=lambda x: x[::-1])

        ddetect_paths = [ 
            i for i in train_paths if int(i.split('.')[0][-1]) in [1]
            ]
        
        train_dataset = kadid10k(train_paths)
        test_dataset = kadid10k(test_paths)
        drift_dataset = kadid10k(ddetect_paths)

    else:
        # Example image paths and labels
        image_paths = [
            'drone_factory_frames/frame_0_{}.jpg'.format(i) for i in range(1, 401)]

        # Split dataset
        train_paths, test_paths = train_test_split(
            image_paths, test_size=0.5, random_state=42, shuffle=False)

        train_dataset = VideoFootage(train_paths)
        test_dataset = VideoFootage(test_paths, scenario=scenario, display_im=True)
        drift_dataset = VideoFootage(train_paths, scenario="normal")



    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    ddetector_loader = DataLoader(
        drift_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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

def load_drd(enc: nn.Module, 
             ddetector_dts: DataLoader, 
             dd_type:  str = "mmd", 
             feat_ext_slice: int = -2):

    model = enc.to(device).eval()
    mini = T.Compose([T.Resize((128,128))])

    ddetect = drift_detector(detector=dd_type)

    if feat_ext_slice!=0:
        feat_ext = torch.nn.Sequential(
            *(list(model.model.children())[:feat_ext_slice])).eval().to(device)

    else:
        feat_ext = torch.nn.Sequential(
            *(list(model.model.children())[:])).eval().to(device)
        
    for bim, _, _ in tqdm(ddetector_dts, desc="Drift fit"):
        inp = feat_ext(mini(bim).to(device)).reshape(bim.shape[0], -1)
        ddetect.fit(inp)

    return feat_ext, ddetect

def compute_quality_score(model, img):
    """Compute the quality score of the image."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, return_embedding=False, scale_score=True)

    return score

if __name__ == "__main__":
    mini = T.Compose([T.Resize((128,128))])
    dataset="kadid10k"
    global_batch_size = 32
    train, test, ddet = init_dataloaders(
        dataset=dataset, batch_size=global_batch_size)

    model_arniqa = load_arniqa_model().to(device)

    model_drd, ddetect = load_drd(
        enc=deepcopy(model_arniqa.encoder),
        ddetector_dts=ddet,
        feat_ext_slice=-2, 
        )
     
    model_drd.to(device)

    all_drift_p_values = []
    all_iqscore_values = []

    drift_pred = []
    drift_tar = []

    poor_quality_pred = []
    poor_quality_tar = []

    with torch.no_grad():
        im_passed = []; idx=0
        for bimages, _, labels in tqdm(test, desc="Test", position=0):

            arniqa_outputs = compute_quality_score(
                model_arniqa, bimages.to(device),
                )
            bimages = mini(bimages)
            dd_in = model_drd(bimages.to(device)).reshape(bimages.shape[0], -1)
            pv = ddetect.forward(dd_in).item()


            meaniq = arniqa_outputs.mean().item()

            all_drift_p_values.extend([pv])
            all_iqscore_values.extend([meaniq])

            logger.info(("drift p-val:",pv , "mean_iq:", meaniq))

            # CALCULATE STATISTICS FOR DRIFT
            if pv>0.05:
                drift_pred.append(0)
            else:
                drift_pred.append(1)

            ideal_labels = torch.tensor([1]*bimages.shape[0])

            if torch.eq(ideal_labels,labels).sum() < bimages.shape[0]//2:
                drift_tar.append(1)
            else:
                drift_tar.append(0)

            # CALCULATE STATISTICS FOR IQA
            for ao, lbl in zip(arniqa_outputs, labels):
                if ao.item()>=0.5:
                    poor_quality_pred.append(0)
                else:
                    poor_quality_pred.append(1)

                if lbl.cpu().item()==1:
                    poor_quality_tar.append(0)
                else:
                    poor_quality_tar.append(1)


logger.info("\nDrift stats")
logger.info("Precision:%f",precision_score(drift_tar, drift_pred))
logger.info("Recall:%f",recall_score(drift_tar, drift_pred))
logger.info("F1:%f",f1_score(drift_tar, drift_pred))
logger.info("\nIQA stats")
logger.info("Precision:%f",precision_score(poor_quality_tar, poor_quality_pred))
logger.info("Recall:%f",recall_score(poor_quality_tar, poor_quality_pred))
logger.info("F1:%f",f1_score(poor_quality_tar, poor_quality_pred))

plt.subplot(2, 1, 1)
plt.plot(all_drift_p_values, marker = 'o')
plt.axvline(len(test)/2, linestyle="--", color="grey")
plt.axhline(0.05, linestyle="--", color="red")
plt.ylabel("drift p-value")
plt.grid()

plt.subplot(2, 1, 2)
plt.axvline(len(test)/2, linestyle="--", color="grey")
plt.axhline(0.5, linestyle="--", color="red")
plt.plot(all_iqscore_values, color="purple", marker = 'o')
plt.ylabel("mean image quality score")
plt.xlabel("# of batches")
plt.grid()

plt.savefig("current_results.jpg")

# with open('drifts.csv', 'w', newline='') as csvfile:
#     drift_values = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for res, step in zip(all_drift_p_values, im_passed):
#         drift_values.writerow([step, res])

# with open('iqs.csv', 'w', newline='') as csvfile:
#     iqa_values = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for res, step in zip(all_iqscore_values, im_passed):
#         iqa_values.writerow([step, res])

# with open('class_choice.csv', 'w', newline='') as csvfile:
#     class_ans = csv.writer(csvfile,
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for cl in class_answers:
#         class_ans.writerow([cl])

# def load_drd(train_dts: DataLoader, 
#     ddetector_dts: DataLoader,
#     model_type: str = "resnet50",  
#     num_epochs: int = 100, 
#     detector: str = "mmd", 
#     dataset: str = "kadid10k", 
#     feat_ext_slice: int = -2,
#     emb_dim: int = 128,
#     dd_type: str = "mmd"):

#     if not os.path.exists("drd_{}_{}.pth".format(model_type, dataset)):
#         model = ResNet(
#             embedding_dim=emb_dim, model=model_type, use_norm=False).to(device)
#         ddetect = drift_detector(detector=detector)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#         for _ in tqdm(range(num_epochs), desc="Epoch", position=0):
#             model.train()
#             running_loss = 0.0
#             for _, images, labels in tqdm(train_dts,desc="#b",position=1,leave=False):
#                 optimizer.zero_grad()
#                 _, outputs = model(images.to(device))
#                 loss = criterion(outputs.to(device), labels.to(device))
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()
#             logger.info(f"Loss: {running_loss/len(train_dts):.4f}")

#         torch.save(model.state_dict(), "drd_{}_{}.pth".format(model_type, dataset))

#     else:
#         logger.info("load from dir")
#         model = ResNet(
#             embedding_dim=emb_dim, model=model_type, 
#             use_norm=False).to(device)
#         model.load_state_dict(
#             torch.load("drd_{}_{}.pth".format(model_type, dataset))
#             )
#         model = model.to(device)

#     ddetect = drift_detector(detector=dd_type)

#     if feat_ext_slice!=0:
#         feat_ext = torch.nn.Sequential(
#             *(list(model.model.children())[:feat_ext_slice])).eval().to(device)

#     else:
#         feat_ext = torch.nn.Sequential(
#             *(list(model.model.children())[:])).eval().to(device)
        
#     for _, im, _ in tqdm(ddetector_dts, desc="Drift fit"):  
#         inp = feat_ext(im.to(device)).reshape(im.shape[0], -1)
#         ddetect.fit(inp)

#     return feat_ext, ddetect