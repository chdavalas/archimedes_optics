from torchdrift.detectors import KernelMMDDriftDetector, KSDriftDetector
import torch
import torch.nn as nn

class drift_detector(nn.Module):
    def __init__(self, detector="mmd") -> None:
        super(drift_detector, self).__init__()
        if detector=="mmd":
            print(detector)
            self.detector = KernelMMDDriftDetector(return_p_value=True)
        else:
            self.detector = KSDriftDetector(return_p_value=True)

    def fit(self, inp):
        self.detector.fit(inp)

    def forward(self, y_pred):
        return self.detector.forward(y_pred)