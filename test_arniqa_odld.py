
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
        urllib.request.urlretrieve(url, file_name)
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
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

    global_batch_size = 32

    # Split dataset
    train_paths, test_paths = train_test_split(
        image_paths, test_size=0.25, random_state=42, shuffle=False)

    # Create datasets and loaders
    train_dataset = kadid10k(train_paths)
    test_dataset = kadid10k(test_paths)

    train_loader = DataLoader(train_dataset, batch_size=global_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=global_batch_size, shuffle=False, drop_last=True)


    return train_loader, test_loader


def load_arniqa_model():
    """Load the pre-trained model."""
    model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                           regressor_dataset="kadid10k")  # You can choose any of the available datasets

    return model

def load_odld(train_dts, num_epochs=15, b_size=32):
    if not os.path.exists("odld.pth"):
        model = resnet50().to(device)
        ddetect = drift_detector()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
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
        base_class = torch.tensor(1. , dtype=torch.float32)
        model = resnet50()
        model.load_state_dict(torch.load("odld.pth"))

        
    ddetect = drift_detector()
    
    for _, _, lbl in tqdm(train_dts):
        ddetect.fit(lbl.shape[0])

    return model.eval(), ddetect




def compute_quality_score(model, img, img_ds):
    """Compute the quality score of the image."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)

    return score

if __name__ == "__main__":
    train, test = init_dataloaders()

    global_batch_size = 64

    model_arniqa = load_arniqa_model()
    model_odld, ddetect = load_odld(train_dts=train, b_size=global_batch_size)

    model_odld.to(device)

    all_labels = []
    all_preds = []
    all_drift_p_values = []
    all_iqscore_values = []
    class_array = np.array(["1.0", "5.0"])

    frames_images = []
    frames_results = []

    class_answers = []


    with torch.no_grad():
        im_passed = []; idx=0
        for bimages, images, labels in tqdm(test, desc="Test", position=0):
            
            outputs = model_odld(images.to(device))
            arniqa_outputs = compute_quality_score(model_arniqa, bimages, images)

            # ddetect.fit(bimages.shape[0])
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            preds = preds.unsqueeze(1).to(dtype=torch.long)
            pv = ddetect.forward(preds.cpu())
            all_drift_p_values.extend([pv.item()])

            print(pv.item(), arniqa_outputs.mean().item())

            all_iqscore_values.extend([arniqa_outputs.mean().item()])

            idx+=images.shape[0]; im_passed += [idx]
            for lb in labels:
                class_answers += [lb.numpy()]

# logger.info classification report
logger.info(classification_report(
    all_labels, 
    all_preds, 
    target_names=class_array)
    )