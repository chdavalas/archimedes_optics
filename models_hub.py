
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, resnet50
from losses import nt_xent_loss
from typing import Tuple
# import torchvision.transforms.v2 as T
import torchvision.transforms as T

dependencies = ["torch"]

available_datasets = ["live", "csiq", "tid2013", "kadid10k", "flive", "spaq", "clive", "koniq10k"]

available_datasets_ranges = {
    "live": (1, 100),
    "csiq": (0, 1),
    "tid2013": (0, 9),
    "kadid10k": (1, 5),
    "flive": (1, 100),
    "spaq": (1, 100),
    "clive": (1, 100),
    "koniq10k": (1, 100),
}

available_datasets_mos_types = {
    "live": "dmos",
    "csiq": "dmos",
    "tid2013": "mos",
    "kadid10k": "mos",
    "flive": "mos",
    "spaq": "mos",
    "clive": "mos",
    "koniq10k": "mos",
}

base_url = "https://github.com/miccunifi/ARNIQA/releases/download/weights"

class Net(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*14*14, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, emb_dim)

    def forward(self, x):
        x = T.Resize((128,128))(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleClassifier(nn.Module):

    def __init__(self, out_classes :int = 25):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 64* 8, 128)
        self.fc2 = nn.Linear(128, out_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 64* 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet18(nn.Module):
    """
    ResNet model with a projection head.

    Args:
        embedding_dim (int): embedding dimension of the projection head
        pretrained (bool): whether to use pretrained weights
        use_norm (bool): whether to normalize the embeddings
    """
    def __init__(self,  
                 head_dim: int, 
                 pretrained: bool = True, 
                 use_norm: bool = True):
        
        super(ResNet18, self).__init__()

        self.pretrained = pretrained
        self.use_norm = use_norm
        self.head = head_dim

        if self.pretrained:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1  
        else:
            weights = None
        self.model = resnet18(weights=weights)

        self.feat_dim = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.head = nn.Sequential(nn.Linear(self.feat_dim, self.head))

    def forward(self, x):
        # x = T.Resize((224,224))(x)
        f = self.model(x)
        g = self.head(f.squeeze())
        return g

# Mostly taken from https://github.com/miccunifi/ARNIQA
class ResNet(nn.Module):
    """
    ResNet model with a projection head.

    Args:
        embedding_dim (int): embedding dimension of the projection head
        pretrained (bool): whether to use pretrained weights
        use_norm (bool): whether to normalize the embeddings
    """
    def __init__(self,  
                 embedding_dim: int = 128, 
                 pretrained: bool = True, 
                 use_norm: bool = True):
        
        super(ResNet, self).__init__()

        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        if self.pretrained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1  
        else:
            weights = None
        self.model = resnet50(weights=weights)

        self.feat_dim = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.embedding_dim)
        )

    def forward(self, x):
        f = self.model(x)
        f = f.view(-1, self.feat_dim)

        if self.use_norm:
            f = F.normalize(f, dim=1)

        g = self.projector(f)
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g


# Mostly taken from https://github.com/miccunifi/ARNIQA
class ARNIQA(nn.Module):
    """
    ARNIQA model for No-Reference Image Quality Assessment (NR-IQA). It is composed of a ResNet-50 encoder and a Ridge
    regressor. The regressor is trained on the dataset specified by the parameter 'regressor_dataset'. The model takes
    in input an image both at full-scale and half-scale. The output is the predicted quality score. By default, the
    predicted quality scores are in the range [0, 1], where higher is better. In addition to the score, the forward
    function allows returning the concatenated embeddings of the image at full-scale and half-scale. Also, the model can
    return the unscaled score (i.e. in the range of the training dataset).
    """
    def __init__(self, regressor_dataset: str = "kadid10k"):
        super(ARNIQA, self).__init__()
        assert regressor_dataset in available_datasets, f"parameter training_dataset must be in {available_datasets}"
        self.regressor_dataset = regressor_dataset

        self.encoder = ResNet(embedding_dim=128, pretrained=True, use_norm=True)
        self.encoder.load_state_dict(torch.hub.load_state_dict_from_url(f"{base_url}/ARNIQA.pth", progress=True,
                                                                    map_location="cpu"))

        self.encoder.eval()
        self.regressor: nn.Module = torch.hub.load_state_dict_from_url(f"{base_url}/regressor_{regressor_dataset}.pth",
                                                                        progress=True, map_location="cpu")
        self.regressor.eval()
        self.mini = T.Compose([T.Resize((128,128))])

    def forward(self, img, return_embedding: bool = False, scale_score: bool = True):
        
        f, _ = self.encoder(img)
        img_ds = self.mini(img) 
        f_ds, _ = self.encoder(img_ds)
        f_combined = torch.hstack((f, f_ds))
        score = self.regressor(f_combined)
        if scale_score:
            score = self._scale_score(score)
        if return_embedding:
            return score, f_combined
        else:
            return score

    def _scale_score(self, score: float, new_range: Tuple[float, float] = (0., 1.)) -> float:
        """
        Scale the score in the range [0, 1], where higher is better.

        Args:
            score (float): score to scale
            new_range (Tuple[float, float]): new range of the scores
        """

        # Compute scaling factors
        original_range = (available_datasets_ranges[self.regressor_dataset][0], available_datasets_ranges[self.regressor_dataset][1])
        original_width = original_range[1] - original_range[0]
        new_width = new_range[1] - new_range[0]
        scaling_factor = new_width / original_width

        # Scale score
        scaled_score = new_range[0] + (score - original_range[0]) * scaling_factor

        # Invert the scale if needed
        if available_datasets_mos_types[self.regressor_dataset] == "dmos":
            scaled_score = new_range[1] - scaled_score

        return scaled_score


class LSTM_drift(nn.Module):
    def __init__(self, 
                 emb_size: int = 128, 
                 hid_size: int = 50, 
                 num_layers: int=2, 
                 class_out_size: int=1):
        
        super(LSTM_drift, self).__init__()
        
        self.encoder = ResNet(
            embedding_dim=emb_size, 
            pretrained=True, 
            use_norm=True)
        
        self.encoder.load_state_dict(
            torch.hub.load_state_dict_from_url(
                f"{base_url}/ARNIQA.pth", progress=True,
                map_location="cpu")
                )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        self.lstm = nn.LSTM(
            input_size=emb_size, 
            hidden_size=hid_size, 
            num_layers=num_layers, 
            batch_first=True)
        
        self.linear = nn.Linear(hid_size, class_out_size)

    def forward(self, x):
        _, x = self.encoder(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

# Mostly taken from https://github.com/miccunifi/ARNIQA
# class SimCLR(nn.Module):
#     """
#     SimCLR model class used for pre-training the encoder for IQA.

#     Args:
#         encoder_params (dict): encoder parameters with keys
#             - embedding_dim (int): embedding dimension of the encoder projection head
#             - pretrained (bool): whether to use pretrained weights for the encoder
#             - use_norm (bool): whether normalize the embeddings
#         temperature (float): temperature for the loss function. Default: 0.1

#     Returns:
#         if training:
#             loss (torch.Tensor): loss value
#         if not training:
#             q (torch.Tensor): image embeddings before the projection head (NxC)
#             proj_q (torch.Tensor): image embeddings after the projection head (NxC)

#     """

#     def __init__(self, encoder_params: DotMap, temperature: float = 0.1, model: str = "50"):
#         super().__init__()

#         self.encoder = ResNet(embedding_dim=encoder_params.embedding_dim,
#                               pretrained=encoder_params.pretrained,
#                               use_norm=encoder_params.use_norm,
#                               model=model)

#         self.temperature = temperature
#         self.criterion = nt_xent_loss

#     def forward(self, im_q, im_k=None):
#         q, proj_q = self.encoder(im_q)

#         if not self.training:
#             return q, proj_q

#         k, proj_k = self.encoder(im_k)
#         loss = self.criterion(proj_q, proj_k, self.temperature)
#         return loss
