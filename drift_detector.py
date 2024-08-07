from torchdrift.detectors import KernelMMDDriftDetector, KSDriftDetector
import torch
import torch.nn as nn
from alibi_detect.cd import MMDDrift
from numpy import concatenate
import torchvision.transforms.v2 as T

class drift_detector(nn.Module):
    def __init__(self, detector="mmd") -> None:
        super(drift_detector, self).__init__()
        self.detector = None
        # if detector=="mmd":
        #     self.detector = KernelMMDDriftDetector(return_p_value=True)
        # else:
        #     self.detector = KSDriftDetector(return_p_value=True)

    def fit(self, inp):
        self.detector = MMDDrift(inp.cpu().numpy(), backend="pytorch", device='cuda',)
        return self.detector

    def forward(self, y_pred):
        print(self.detector.predict(y_pred.cpu().numpy())["data"])
        return self.detector.predict(y_pred.cpu().numpy())["data"]["distance"]