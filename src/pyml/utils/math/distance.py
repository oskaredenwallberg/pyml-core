import numpy as np

# -------

def euclidean_distance(
        a1: np.ndarray, 
        a2: np.ndarray, 
        axis: int = 1,
        ) -> np.ndarray:
    """
    Calculate the Euclidean distance between two points p1 & p2.
    
    Parameters
    ----------
    a1 : ndarray
        A single datapoint or an array of datapoints. a1 must always have the same 
        number of dimensions as a2, and the same shape as a2 if an array is passed. 
    a2 : ndarray
        A single datapoint or an array of datapoints. 
    axis : int, default = 1
        The axis to sum the dimensions across. 

    Returns
    -------
    ndarray
        The euclidean distances between the datapoints. One-to-many or many-to-many. 

    Examples
    --------
    # One to many
    >>> a1 = np.ndarray([1])
    >>> a2 = np.ndarray([3, 4, 2])
    >>> distances = euclidean_distances()
    """
    assert a2.ndim == a1.ndim, \
        f"Error: Incompatible dimensions a1 {a1.ndim}, a2 {a2.ndim}"
    assert a2.shape == a1.shape or \
        a1.shape[0] == 1 or \
        a2.shape[0] == 1, \
        f"Error: Incompatible shapes a1 {a1.shape}, a2 {a2.shape}. "

    # Decide the order when taking the difference
    if a2.shape[0] >= a1.shape[0]:
        squared_differences = np.square(a2-a1)
    else:
        squared_differences = np.square(a1-a2)
    
    # Check for multiple dimensions to sum across before root
    if squared_differences.ndim > 1:  
        squared_differences = squared_differences.sum(axis=axis)

    # Final root
    distance = np.sqrt(squared_differences)

    return distance
