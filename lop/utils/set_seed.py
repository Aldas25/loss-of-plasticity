import numpy as np
import torch

def get_random_int():
    """
    Generates a random integer between 0 and 1e9.
    """
    return np.random.randint(0, 10**9)

# Setting seed for result reproducibility.
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)