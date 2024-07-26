import numpy as np
import torch
from typing import List
import torch.nn.functional as F

class GaussianNoise:

    def __init__(self, sigma: float):
        """Create noise distribution.

        Args:
            sigma (float): Noise scaling.
        """
        self.sigma = sigma

    def sample(self, shape: List[int]) -> torch.FloatTensor:
        """Sample from noise distribution.

        Args:
            shape (list[int]): Shape of samples.

        Returns:
            torch.FloatTensor: Noise samples.
        """
        return self.sigma * torch.randn(*shape)

    def kernel(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Apply kernel from noise distribution.

        Args:
            x (torch.FloatTensor): Input tensor to apply kernel to.

        Returns:
            torch.FloatTensor: Resulting tensor.
        """
        return (
            1
            / (self.sigma * np.sqrt(2 * np.pi))
            * torch.exp(-torch.square(x) / (2 * self.sigma**2))
        )

def inv_sigmoid(x: torch.FloatTensor) -> torch.FloatTensor:
    """Inverse sigmoid.

    Args:
        x (torch.FloatTensor): Torch tensor to apply to.

    Returns:
        torch.FloatTensor: x after applying inverse sigmoid.
    """
    return torch.log(x) - torch.log(1 - x)


def kernel_reg(logits: torch.FloatTensor, labels: torch.LongTensor, ts: torch.FloatTensor, noise) -> torch.FloatTensor:
    """Computes kernel regression using provided noise distribition.

    Args:
        logits (torch.FloatTensor): Logits tensor.
        labels (torch.LongTensor): Labels tensor.
        ts (torch.FloatTensor): Tensor of noised logits.
        noise: Should be either GaussianNoise or UniformNoise.

    Returns:
        torch.FloatTensor: Returns estimate of conditional expectation.
    """
    total = noise.kernel(ts.unsqueeze(dim=1) - logits)
    return (total * labels.unsqueeze(dim=0)).mean(dim=1) / total.mean(dim=1)


def logit_smoothed_ece(logits: torch.FloatTensor, labels: torch.LongTensor) -> float:
    """Computes logit smoothed ECE.

    Args:
        logits (torch.FloatTensor): Logits tensor.
        labels (torch.LongTensor): Labels tensor.
        ts (torch.FloatTensor): Tensor of noised logits.
        noise: Should be either GaussianNoise or UniformNoise.
        reduce (bool, optional): Whether to return reduced value or not. Defaults to True.

    Returns:
        float: Logit-smoothed ECE value, returned if reduce=True. 
    """
    # Expects logits to be shape (n, 1) and labels to be shape (n, 1).
    n_t = len(logits)

    if isinstance(logits,np.ndarray):
        logits = torch.from_numpy(logits)

    if isinstance(labels,np.ndarray):
        labels = torch.from_numpy(labels)
    
    noise = GaussianNoise(sigma=1/15)

    emp_sample = torch.randint(len(logits), (n_t,))
    ts = logits[emp_sample] + noise.sample((n_t,1))
    ests = kernel_reg(logits, labels, ts, noise)
    return torch.abs((ests - torch.sigmoid(ts))).mean()

if __name__=="__main__":
    import os
    import sys
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,parentdir) 
    from tools.Calibration.utils import Load_z
    from customKing.config.config import get_cfg

    task_mode = "Calibration"
    cfg = get_cfg(task_mode)
    testDataset = Load_z(cfg,"valid")
    z_list,label_list = testDataset.z_list,testDataset.label_list
    max_z_list, predictions = torch.max(z_list, 1)
    accs = predictions.eq(label_list)
    LS_ECE = logit_smoothed_ece(max_z_list,accs)
    print(LS_ECE)