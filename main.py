
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
from torchvision.models import resnet18
from sklearn.metrics import precision_score, recall_score, f1_score

from models_hub import ResNet, ARNIQA, LSTM_drift, Net, ResNet18
import csv
import matplotlib.pyplot as plt
from dotmap import DotMap
from copy import deepcopy
from random import shuffle

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='output.log', 
    filemode='w', 
    format='%(levelname)s:%(message)s',
    level=logging.INFO)

torch.manual_seed(0)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def init_dataloaders(dataset="kadid10k", batch_size=32):

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
        
        test_paths = sorted(test_paths, key=lambda x: x[::-1])

        train_dataset = kadid10k(train_paths)
        test_dataset = kadid10k(test_paths)

        # held-out from training+testing
        drift_dataset = kadid10k(pristine_images)

    else:
        # Example image paths and labels (we take the images sorted very fast)
        im_count = len(os.listdir(dataset))
        image_paths = [
            dataset+'/frame_'+dataset+'_{}.jpg'.format(i) for i in range(im_count)]
        

        # Split dataset
        train_paths, test_paths = train_test_split(
            image_paths, test_size=0.25, random_state=42, shuffle=False)

        _, dd_paths = train_test_split(
            train_paths, test_size=0.5, random_state=42, shuffle=True)
    

        train_dataset = VideoFootage(train_paths, distort="rand")
        test_dataset = VideoFootage(test_paths, distort="last")
        drift_dataset = VideoFootage(dd_paths)
        print(train_dataset.__len__())
        print(test_dataset.__len__())
        print(drift_dataset.__len__())


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

def load_drd(ddetector_dts: DataLoader, 
             train_dts: DataLoader,
             dd_type:  str = "mmd", 
             feat_ext_slice: int = 3,
             num_epochs: int = 30,
             emb_dim: int = 10):

    model_type = "simple"
    # model = enc.to(device).eval()

    ddetect = drift_detector(detector=dd_type)
    model = ARNIQA().encoder
    # if not os.path.exists("drd_{}.pth".format(model_type)):
    #     model = ResNet18(head_dim=emb_dim).to(device)

    #     torch.save(model.state_dict(), "drd_{}.pth".format(model_type))
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(model.parameters(), lr=0.001)
    #     for _ in tqdm(range(num_epochs), desc="Epoch", position=0):
    #         model.train()
    #         running_loss = 0.0
    #         for bimages, labels in tqdm(train_dts,desc="#b",position=1,leave=False):

    #             outputs = model(bimages.to(device))
    #             # labels = torch.nn.functional.one_hot(labels, emb_dim)
    #             loss = criterion(outputs.to(device), labels.long().to(device))
    #             loss.backward()
    #             optimizer.step()
    #             running_loss += loss.item()
    #             #print(torch.sum(outputs.argmax(dim=1).to(device)==labels.to(device))/bimages.shape[0])

    #         logger.info(f"Loss: {running_loss/len(train_dts):.4f}")
    #         # print(outputs, labels)
    #         torch.save(model.state_dict(), "drd_{}.pth".format(model_type))

    # else:
    #     logger.info("load from dir")
    #     model = ResNet18(head_dim=emb_dim)
    #     model.load_state_dict(
    #         torch.load("drd_{}.pth".format(model_type))
    #         )
    model = model.to(device)
    
    if feat_ext_slice!=0:
        feat_ext = torch.nn.Sequential(
            *(list(model.model.children())[:feat_ext_slice])).eval().to(device)

    else:
        feat_ext = deepcopy(model.model).eval().to(device)
    
    
    for param in feat_ext.parameters():
        param.requires_grad = False

    for bim, _ in tqdm(ddetector_dts, desc="Drift fit"):
        bim =  T.FiveCrop(size=360)(bim)
        bim  = torch.cat(bim, dim=0)
        inp = feat_ext(bim.to(device))
        inp = inp.reshape(bim.shape[0], -1)
        ddetect.fit(inp)

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
    """Compute the quality score of the image."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, return_embedding=False, scale_score=True)

    return score

if __name__ == "__main__":
    
    dataset="traffic_inspection"
    # dataset="pipe_inspection"
    # dataset="factory_inspection"
    # dataset="assembly_line_extreme_inspection"
    # dataset="assembly_line_inspection"
    # dataset="kadid10k"
    global_batch_size = 32
    train, test, ddet = init_dataloaders(
        dataset=dataset, batch_size=global_batch_size)

    model_arniqa = load_arniqa_model().to(device)

    model_drd, ddetect = load_drd(
        ddetector_dts=ddet,
        train_dts=train,
        feat_ext_slice=0, 
        num_epochs=10,
        emb_dim=3
    )
     
    # ONLY TRAINING FOR THE MOMENT (TRAINING NEEDS CHANGES TOO)
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


    with torch.no_grad():
        im_passed = []; idx=0
        for bimages, labels in tqdm(test, desc="Test", position=0):
            
            lstm_outputs = model_lstm(bimages.to(device))
            
            arniqa_outputs = compute_quality_score(
                model_arniqa, bimages.to(device),
                )
            bim = T.FiveCrop(size=360)(bimages)
            bim  = torch.cat(bim, dim=0)
            dd_in = model_drd(bim.to(device))
            dd_in = dd_in.reshape(bim.shape[0], -1)
            pv = ddetect.forward(dd_in).item()
            if pv>0:
                print("NO drift detected:",pv)
                # input()
 
            meaniq = arniqa_outputs.mean().item()
            lstm_labels = torch.where(labels>1, 1, 0)
            lstm_labels = torch.nn.functional.one_hot(lstm_labels.long(), 2)
            lstm_mean = torch.argmax(lstm_outputs, dim=1).float().mean()

            # print(dd_in, arniqa_outputs, dd_in.argmax(dim=1), labels)

            all_drift_p_values.append(pv)
            mean_iqscore_values.append(meaniq)
            all_lstm_mean_values.append(lstm_mean.item())

            logger.info("---------")
            logger.info(("drift p-val:",pv))
            logger.info(("mean_iq:", meaniq))
            logger.info(("lstm_mean:", lstm_mean.item()))

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

            # CALCULATE STATISTICS FOR LSTM
            if lstm_mean.item()>0.5:
                lstm_drift_pred.append(1)
            else:
                lstm_drift_pred.append(0)

            lstm_labels_argmax = torch.argmax(lstm_labels, dim=1)
            lstm_labels_01 = torch.where(labels>1, 1, 0)
            if torch.eq(lstm_labels_argmax,lstm_labels_01).sum() > bimages.shape[0]//2:
                lstm_drift_tar.append(1)
            else:
                lstm_drift_tar.append(0)

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

logger.info("--------------------------------------------------")
logger.info("Drift stats")
logger.info("Precision:%f",precision_score(drift_tar, drift_pred))
logger.info("Recall:%f",recall_score(drift_tar, drift_pred))
logger.info("F1:%f",f1_score(drift_tar, drift_pred))
logger.info("--------------------------------------------------")
logger.info("IQA stats")
logger.info("Precision:%f",precision_score(poor_quality_tar, poor_quality_pred))
logger.info("Recall:%f",recall_score(poor_quality_tar, poor_quality_pred))
logger.info("F1:%f",f1_score(poor_quality_tar, poor_quality_pred))
logger.info("--------------------------------------------------")
logger.info("LSTM stats")
logger.info("Precision:%f",precision_score(lstm_drift_tar, lstm_drift_pred))
logger.info("Recall:%f",recall_score(lstm_drift_tar, lstm_drift_pred))
logger.info("F1:%f",f1_score(lstm_drift_tar, lstm_drift_pred))
logger.info("--------------------------------------------------")



plt.subplot(3, 1, 1)
plt.title(dataset)
plt.plot(all_drift_p_values)
plt.axvline(len(test)/2, linestyle="--", color="grey")
plt.axhline(0.05, linestyle="--", color="red")
x  = [ i for i, _ in enumerate(all_drift_p_values)]
plt.fill_between(x, 0, 0.05, alpha=0.3, color="red")
plt.ylabel("drift p-value")
plt.grid()

plt.subplot(3, 1, 2)
plt.axvline(len(test)/2, linestyle="--", color="grey")
plt.axhline(0.5, linestyle="--", color="red")
x  = [ i for i, _ in enumerate(all_drift_p_values)]
plt.fill_between(x, 0, 0.5, alpha=0.3, color="red")
plt.plot(mean_iqscore_values, color="purple")
plt.ylabel("mean image\nquality score")
plt.grid()

plt.subplot(3, 1, 3)
plt.axvline(len(test)/2, linestyle="--", color="grey")
plt.axhline(0.5, linestyle="--", color="red")
x  = [ i for i, _ in enumerate(all_drift_p_values)]
plt.fill_between(x, 0.5, 1, alpha=0.3, color="red")
plt.plot(all_lstm_mean_values, color="blue")
plt.ylabel("drift detected\nlstm score")
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