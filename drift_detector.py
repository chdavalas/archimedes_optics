from torchdrift.detectors import KernelMMDDriftDetector, KSDriftDetector
import torch
import torch.nn as nn
from alibi_detect.cd import MMDDrift
from numpy import concatenate
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='output.log', 
    filemode='w', 
    format='%(levelname)s:%(message)s',
    level=logging.INFO)

class drift_detector(nn.Module):
    def __init__(self, detector="mmd") -> None:
        super(drift_detector, self).__init__()
        self.detector = None
        # if detector=="mmd":
        #     self.detector = KernelMMDDriftDetector(return_p_value=True)
        # else:
        #     self.detector = KSDriftDetector(return_p_value=True)

    def fit(self, inp):
        self.detector = MMDDrift(inp.cpu().numpy(), backend="pytorch", device='cuda')
        return self.detector

    def forward(self, y_pred):
        results = self.detector.predict(y_pred.cpu().numpy())["data"]
        logger.info("drift distance/threshold: {}/{}".format(results["distance"], results["distance_threshold"]))
        return results["p_val"]