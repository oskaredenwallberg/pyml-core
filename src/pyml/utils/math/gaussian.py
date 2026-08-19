import numpy as np
from typing import Callable

def get_gaussian_pdf(mean: np.number, std: np.number) -> Callable[[np.number], np.float32]:
    """
    Retrieve gaussian 
    """
    var = std**2
    return lambda x: 1/np.sqrt(2*np.pi*var) * np.exp(-(x - mean)**2 / (2*var))
