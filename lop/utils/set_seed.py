import numpy as np
import torch

# Setting seed for result reproducibility.
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)