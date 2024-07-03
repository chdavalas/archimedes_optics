from torchdrift.detectors import KernelMMDDriftDetector, Detector, KSDriftDetector
import torch
import torch.nn as nn

class drift_detector(nn.Module):
    def __init__(self) -> None:
        super(drift_detector, self).__init__()
        self.detector = KernelMMDDriftDetector(return_p_value=True)

    def fit(self, inp, b_size=32):
        self.detector.fit(inp.long())

    def forward(self, y_pred):
        return self.detector.compute_p_value(y_pred)