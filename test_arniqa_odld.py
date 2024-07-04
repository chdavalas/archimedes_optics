
import urllib.request
import os
import torch
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataset_kadid10k import kadid10k
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from drift_detectors import drift_detector
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import logging
from torchvision.models import resnet50
import numpy as np
from sklearn.metrics import classification_report
from models_dist import DistortionClassifier

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='output.log', 
    filemode='w', 
    format='%(levelname)s:%(message)s',
    level=logging.INFO)

torch.manual_seed(0)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def init_dataloaders():

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

    global_batch_size = 32

    # Split dataset
    train_paths, test_paths = train_test_split(
        image_paths, test_size=0.4, random_state=42, shuffle=True)

    prist_train_paths, prist_test_paths = train_test_split(
        pristine_images, test_size=0.4, random_state=42, shuffle=True)

    train_paths += prist_train_paths
    test_paths = prist_test_paths + sorted(test_paths, key=lambda x: x[::-1])

    ddetect_paths = [ i for i in train_paths if int(i.split('.')[0][-1]) in [1,2,3]]

    # Create datasets and loaders
    train_dataset = kadid10k(train_paths)
    test_dataset = kadid10k(test_paths)
    drift_dataset = kadid10k(ddetect_paths)

    train_loader = DataLoader(train_dataset, batch_size=global_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=global_batch_size, shuffle=False, drop_last=True)
    ddetector_loader = DataLoader(drift_dataset, batch_size=global_batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader, ddetector_loader


def load_arniqa_model():
    """Load the pre-trained model."""
    model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                           regressor_dataset="kadid10k")  # You can choose any of the available datasets

    return model

def load_odld(train_dts, ddetector_dts, num_epochs=100, b_size=32):

    base_class = torch.tensor(1. , dtype=torch.float32)



    if not os.path.exists("odld.pth"):
        model = resnet50(weights='DEFAULT').to(device)
        ddetect = drift_detector()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        for epoch in tqdm(range(num_epochs), desc="Epoch", position=0):
            model.train()
            running_loss = 0.0
            for bimages, images, labels in tqdm(train_dts, desc="batch", position=1, leave=False):
                optimizer.zero_grad()
                outputs = model(images.to(device))
                loss = criterion(outputs.to(device), labels.long().to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                ddetect.fit(images.shape[0])
            logger.info(f"Loss: {running_loss/len(train_dts):.4f}")

        torch.save(model.state_dict(), "odld.pth")

    else:
        logger.info("load from dir")
        model = resnet50(weights='DEFAULT')
        model.load_state_dict(torch.load("odld.pth"))

    feat_ext = torch.nn.Sequential(*(list(model.children())[:-2])).eval().to(device)
    ddetect = drift_detector()

    for _, im, _ in tqdm(ddetector_dts, desc="Drift fit"):  
        inp = feat_ext(im.to(device))
        ddetect.fit(inp.reshape(im.shape[0], -1))

    return model.eval(), ddetect, feat_ext




def compute_quality_score(model, img, img_ds):
    """Compute the quality score of the image."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)

    return score

if __name__ == "__main__":
    train, test, ddet = init_dataloaders()

    global_batch_size = 32

    model_arniqa = load_arniqa_model().to(device)
    model_odld, ddetect, feat_ext = load_odld(
        train_dts=train, ddetector_dts=ddet, b_size=global_batch_size, 
        )

    model_odld.to(device)
    feat_ext.to(device)

    all_labels = []
    all_preds = []
    all_drift_p_values = []
    all_iqscore_values = []
    class_array = np.array(["{}.0".format(i) for i in range(1,3)])

    frames_images = []
    frames_results = []

    class_answers = []


    with torch.no_grad():
        for bimages, images, labels in tqdm(test, desc="Test", position=0):
            #ddetect.fit(labels.shape[0])
            outputs = model_odld(images.to(device))
            arniqa_outputs = compute_quality_score(model_arniqa, bimages.to(device), images.to(device))

            # ddetect.fit(bimages.shape[0])
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # preds = preds.unsqueeze(1).to(dtype=torch.long)
            dd_in = feat_ext(images.to(device)).reshape(images.shape[0], -1)
            
            pv = ddetect.forward(dd_in).item()

            meaniq = arniqa_outputs.mean().item()

            all_drift_p_values.extend([pv])
            all_iqscore_values.extend([meaniq])


            logger.info(("drift p-val:",pv , "mean_iq:", meaniq))


# logger.info classification report
logger.info(classification_report(
    all_labels, 
    all_preds, 
    target_names=class_array)
    )


plt.subplot(2, 1, 1)
plt.plot(all_drift_p_values, marker = 'o')
plt.axvline(len(test)/2, linestyle="--", color="grey")
plt.axhline(0.05, linestyle="--", color="red")
plt.ylabel("drift p-value")
plt.grid()

plt.subplot(2, 1, 2)
plt.axvline(len(test)/2, linestyle="--", color="grey")
plt.plot(all_iqscore_values, color="purple", marker = 'o')
plt.ylabel("mean image quality score")
plt.xlabel("# of batches")
plt.grid()

plt.savefig("current_results.jpg")