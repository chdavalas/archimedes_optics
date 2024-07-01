from torchdrift.detectors import KernelMMDDriftDetector, mmd, KSDriftDetector
import torch
import torch.nn as nn

class drift_detector(nn.Module):
    def __init__(self) -> None:
        super(drift_detector, self).__init__()
        self.detector = KSDriftDetector(return_p_value=True)

    def fit(self, b_size=16):
        base_class = torch.tensor(1., dtype=torch.long)
        self.detector.fit(base_class.repeat(b_size,1))

    def forward(self, y_pred, ):
        return self.detector.compute_p_value(y_pred)