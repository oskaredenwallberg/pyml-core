import numpy as np

def l1_norm(coefficients : np.ndarray) -> float:
     """
     ...
     """
     return np.sum(np.abs(coefficients))

def l2_norm(coefficients : np.ndarray) -> float:
     """
     ...
     """
     return np.sum(np.square(coefficients))