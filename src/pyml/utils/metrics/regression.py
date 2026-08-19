import numpy as np


def mse_score(
        y_true : np.ndarray,
        y_pred : np.ndarray,
        root : bool = False,
        ) -> float:
    """
    Compute the mean squared error (MSE) of the predicted targets against the
    true targets.

    MSE is computed as the mean of the squared differences between the true
    targets and the predicted ones. MSE represents how close the target predictions
    are to the true ones. 

    Parameters
    ----------
    y_true : ndarray
        An array of the true targets.
    y_pred : ndarray
        An array of the predicted targets. Must be of the same length as **y_true**. 
    root : bool
        Flag to use root mean squared error (RMSE) instead MSE.

    Returns
    -------
    float
        The mean squared error of the predicted targets.
    
    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1.21, 3.42, 0.80, 0.44, 7.97])
    >>> y_pred = np.array([1.19, 2.98, 0.56, 0.75, 9.01])  # Close predictions
    >>> mse_score(y_true, y_pred)
    0.28586
    >>> y_true = np.array([1.21, 3.42, 0.80, 0.44, 7.97])
    >>> y_pred = np.array([1.21, 3.42, 0.80, 0.44, 7.97])  # Perfect predictions
    >>> mse_score(y_true, y_pred)
    0.0
    >>> y_true = np.array([1.21, 3.42, 0.80, 0.44, 7.97])
    >>> y_pred = np.array([9.58, 0.03, 5.60, 7.01, 2.31])  # Bad predictions
    >>> mse_score(y_true, y_pred)
    35.957899999999995
    """
    mse = np.mean(np.square(y_true - y_pred))
    return np.sqrt(mse) if root else mse


def mae_score(
    y_true : np.ndarray,
    y_pred : np.ndarray,    
    ) -> float:
    """
    Compute the mean absolute error (MAE) of the predicted targets against the
    true targets.

    MAE is computed as the mean of the absolute differences between the true
    targets and the predicted ones. MAE represents how close the target predictions
    are to the true ones. 

    Parameters
    ----------
    y_true : ndarray
        An array of the true targets.
    y_pred : ndarray
        An array of the predicted targets. Must be of the same length as **y_true**. 

    Returns
    -------
    float
        The mean absolute error of the predicted targets.
    
    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1.21, 3.42, 0.80, 0.44, 7.97])
    >>> y_pred = np.array([1.19, 2.98, 0.56, 0.75, 9.01])  # Close predictions
    >>> mae_score(y_true, y_pred)
    0.41
    >>> y_true = np.array([1.21, 3.42, 0.80, 0.44, 7.97])
    >>> y_pred = np.array([1.21, 3.42, 0.80, 0.44, 7.97])  # Perfect predictions
    >>> mae_score(y_true, y_pred)
    0.0
    >>> y_true = np.array([1.21, 3.42, 0.80, 0.44, 7.97])
    >>> y_pred = np.array([9.58, 0.03, 5.60, 7.01, 2.31])  # Bad predictions
    >>> mae_score(y_true, y_pred)
    5.758000000000001
    """
    mae = np.mean(np.abs(y_true - y_pred))
    return mae
