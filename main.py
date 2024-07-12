
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
from sklearn.metrics import classification_report
from models_hub import ResNet, SimCLR
import csv
import matplotlib.pyplot as plt
from dotmap import DotMap

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

        ddetect_paths = [ i for i in train_paths if int(i.split('.')[0][-1]) in [1,2,3]]
        
        train_dataset = kadid10k(train_paths)
        test_dataset = kadid10k(test_paths)
        drift_dataset = kadid10k(ddetect_paths)

    else:
        # Example image paths and labels
        image_paths = ['drone_factory_frames/frame_0_{}.jpg'.format(i) for i in range(1, 401)]

        # Split dataset
        train_paths, test_paths = train_test_split(
            image_paths, test_size=0.5, random_state=42, shuffle=False)

        train_dataset = VideoFootage(train_paths)
        test_dataset = VideoFootage(test_paths, scenario=scenario, display_im=True)
        drift_dataset = VideoFootage(train_paths, scenario="normal")



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    ddetector_loader = DataLoader(drift_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader, ddetector_loader


def load_arniqa_model(replacement_enc: nn.Module = None, 
                      regr_dt: str = "kadid10k", ):
    """Load the pre-trained model."""
    
    # available_datasets = 
    # [
    #   "live", "csiq", "tid2013", "kadid10k", "flive", 
    #   "spaq", "clive", "koniq10k"
    # ]

    model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                        regressor_dataset=regr_dt)    # You can choose any of the available datasets
    if replacement_enc:
        model.encoder = replacement_enc


    return model.eval().to(device)

def load_drd(model_type: str, 
             train_dts: DataLoader, 
             ddetector_dts: DataLoader, 
             num_epochs: int = 100, 
             detector: str = "mmd", 
             dataset: str = "kadid10k", 
             feat_ext_slice: int = -2,
             emb_dim: int = 128):

    if not os.path.exists("drd_{}_{}.pth".format(model_type, dataset)):
        model = ResNet(embedding_dim=emb_dim, model=model_type, use_norm=False).to(device)
        ddetect = drift_detector(detector=detector)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for _ in tqdm(range(num_epochs), desc="Epoch", position=0):
            model.train()
            running_loss = 0.0
            for _, images, labels in tqdm(train_dts, desc="batch", position=1, leave=False):
                optimizer.zero_grad()
                _, outputs = model(images.to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            logger.info(f"Loss: {running_loss/len(train_dts):.4f}")

        torch.save(model.state_dict(), "drd_{}_{}.pth".format(model_type, dataset))

    else:
        logger.info("load from dir")
        model = ResNet(embedding_dim=emb_dim, model=model_type, use_norm=False).to(device)
        model.load_state_dict(
            torch.load("drd_{}_{}.pth".format(model_type, dataset))
            )
        model = model.to(device)

    ddetect = drift_detector()

    if feat_ext_slice!=0:
        feat_ext = torch.nn.Sequential(
            *(list(model.model.children())[:feat_ext_slice])).eval().to(device)

    else:
        feat_ext = torch.nn.Sequential(
            *(list(model.model.children())[:])).eval().to(device)
        
    for _, im, _ in tqdm(ddetector_dts, desc="Drift fit"):  
        inp = feat_ext(im.to(device)).reshape(im.shape[0], -1)
        ddetect.fit(inp)

    return model.eval(), ddetect, feat_ext


    




def compute_quality_score(model, img, img_ds):
    """Compute the quality score of the image."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)

    return score

if __name__ == "__main__":

    dataset="kadid10k"
    scenario="blur"
    global_batch_size = 32
    train, test, ddet = init_dataloaders(
        dataset=dataset, batch_size=global_batch_size, scenario=scenario)



    model_arniqa = load_arniqa_model().to(device)
    model_drd, ddetect, feat_ext = load_drd(detector="mmd", model_type="resnet18",
        train_dts=train, ddetector_dts=ddet, dataset=dataset, feat_ext_slice=-1, emb_dim=2
        )

    
    model_drd.to(device)
    feat_ext.to(device)

    all_labels = []
    all_preds = []
    all_drift_p_values = []
    all_iqscore_values = []


    frames_images = []
    frames_results = []

    class_answers = []
    class_array = []

    with torch.no_grad():
        im_passed = []; idx=0
        for bimages, images, labels in tqdm(test, desc="Test", position=0):

            _, outputs = model_drd(images.to(device))
            arniqa_outputs = compute_quality_score(
                model_arniqa, bimages.to(device), images.to(device)
                )
            
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

            idx+=images.shape[0]; im_passed += [idx]
            for lb in labels:
                class_answers += [lb.numpy()]
            for pred, lbl in zip(preds, labels):
                if int(pred.cpu().numpy()) not in class_array:
                    class_array.append(int(pred.cpu().numpy()))
                if int(lbl.cpu().numpy()) not in class_array:
                    class_array.append(int(lbl.cpu().numpy()))

            logger.info(("drift p-val:",pv , "mean_iq:", meaniq))



# logger.info classification report
class_array = np.array([str(i) for i in class_array])
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