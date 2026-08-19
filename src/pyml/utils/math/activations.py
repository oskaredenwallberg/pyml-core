import numpy as np

linear_f = lambda z: z
sigmoid_f = lambda z: 1 / (1 + np.exp(-z))
tanh_f = lambda z: np.tanh(z)
threshold_f = lambda z: np.greater_equal(z, 0).astype(int)
relu_f = lambda z: np.max(z, 0)

def linear(z: np.ndarray) -> np.ndarray:
    """
    >>> Identity function
    >>> linear(x) = x
    """
    return z

def sigmoid(z: np.ndarray) -> np.ndarray:  # aka logistic function
    """
    >>> sigmoid(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))

def tanh(z: np.ndarray) -> np.ndarray:
    """
    >>> tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    >>> tanh(z) = (e^z - e^-z) / (e^z + e^-z)
    """
    return np.tanh(z)

def threshold(z: np.ndarray) -> np.ndarray:
    """
    >>> relu(z) = max(z, 0)
    """
    return np.greater_equal(z, 0).astype(int)

def relu(z: np.ndarray) -> np.ndarray:
    """
    >>> relu(z) = max(z, 0)
    """
    return np.max(z, 0)
