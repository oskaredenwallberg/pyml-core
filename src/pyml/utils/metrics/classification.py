import numpy as np


def accuracy_score(
        y_true : np.ndarray,
        y_pred : np.ndarray,
        ) -> float:
    """
    Compute the accuracy of the predicted labels against the true labels.

    Accuracy is computed as the number of correctly predicted labels divided
    by the number of class labels. The accuracy measure represents the 
    percentage of correctly predicted labels. Accuracy ranges between [0, 1].

    Parameters
    ----------
    y_true : ndarray
        An array of the true class labels.
    y_pred : ndarray
        An array of the predicted class labels. Must be of the same length as `y_true`. 
    
    Returns
    -------
    float
        The accuracy score of the predicted class labels.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 0, 0, 1, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 0, 1, 0, 0, 1])  # 5 out of 7 correct
    >>> accuracy_score(y_true, y_pred)
    0.7142857142857143
    >>> y_true = np.array([1, 0, 0, 1, 0, 1, 1])
    >>> y_pred = np.array([1, 0, 0, 1, 0, 1, 1])  # 7 out of 7 correct
    >>> accuracy_score(y_true, y_pred)
    1.0
    >>> y_true = np.array([1, 0, 0, 1, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 0, 1, 0, 0])  # 0 out of 7 correct
    >>> accuracy_score(y_true, y_pred)
    0.0
    """
    accuracy = np.mean(y_true == y_pred)
    return accuracy

def bce_score(
        y_true : np.ndarray,
        y_pred : np.ndarray
        ) -> float:
    """
    Compute the binary cross entropy score of the predicted labels against the true labels.

    >>> # The mean negative log likelihood, abbreviated NLL also
    >>> bce = 1/N * sum(y * log(h) + (1-y) * log(1-h))

    Parameters
    ----------
    y_true : ndarray
        An array of the true class labels.
    y_pred : ndarray
        An array of the predicted class labels. Must be of the same length as `y_true`.  

    Returns
    -------
    float
        The binary cross entropy score of the predicted class labels.

    Examples
    --------
    >>> ... to be continued
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # avoid underflow in log expressions
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce