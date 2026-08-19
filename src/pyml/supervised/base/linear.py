import numpy as np
from typing import Literal
from abc import ABC, abstractmethod

from ...utils.math.regularisation import l1_norm, l2_norm
from ...utils.data import shuffle_data, split_data
from ...utils.validation import EarlyStopper

# --------------------- LINEAR MODEL CLASSES ---------------------
# LinearModel       (ABC)

class LinearModel(ABC):
    def __init__(
            self,
            learning_rate : float = 0.01,
            epochs : int = 200,  # num training rounds
            batch_size : int = 32,  # NOTE only relevant if training method is set to 'mbsgd', ignored otherwise
            keep_rest : int = False,  # whether to use rest batch or not
            training_method : Literal['gd', 'sgd', 'mbsgd'] = 'mbsgd',  # mini-batch stochastic graident descent
            patience : int = 5,  # set -1 to ignore early stopping
            train_val_split : float = 0.2,
            verbose : bool = True,
            ) -> None:
        super().__init__()
        # Parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.keep_rest = keep_rest
        self.training_method = training_method
        self.early_stopper = EarlyStopper(patience)
        self.train_val_split = train_val_split
        self.verbose = verbose

        # Attributes
        self._activation = None # set to identity or sigmoid
        self.w = None  # weights, coefficients
        self.b = None  # bias, intercept
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Gradient descent training of model.

        Main training loop for model. Contains logic for data split into
        training and validation data and for training method adjustments. 
        Initializes weights and iteratively trains and evaluates the model
        over specified epochs. 

        Parameters
        ----------
        X : ndarray
            Array of input feature data. Split into training and validation data.
        y : ndarray
            Array of input target data. Split into training and validation parts.
        """
        self.n_features = X.shape[1]
        self.w = np.random.randn(self.n_features)  # normal initialization, could use uniform also...
        self.b = 0
        losses = np.zeros((self.epochs, 2))  # train and validation

        # train val split
        X, y = shuffle_data(X, y)
        X_train, X_val, y_train, y_val = split_data(X, y, self.train_val_split)  # no internal shuffle in split function

        # adjust the batch size for the training method
        if self.training_method == 'gd':
            self.batch_size = X_train.shape[0]  # loss from entire dataset per update
        if self.training_method == 'sgd':
            self.batch_size = 1                 # loss from one sample per update

        # training loop
        for epoch in range(self.epochs):
            train_loss = self.gradient_descent(X_train, y_train)  # average batch loss over epoch
            eval_loss = self.evaluate(X_val, y_val)
            losses[epoch] = train_loss, eval_loss

            if self.verbose:
                print(f'Epoch {epoch} : Train Loss {train_loss:4f};  Eval Loss {eval_loss:4f}')

            # Early stopping
            if self.early_stopper is not None and self.early_stopper.check_stop(eval_loss):
                break
        
        self.early_stopper.reset()  # reset, should the model be retrained
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute target predictions.

        Computes a linear combination between the input data X and the stored
        parameters (coefficients and intercept). 

        Parameters
        ---------
        X : ndarray
            Array of input data to be evaluated.

        Returns
        -------
        ndarray
            Array of target predictions.
        """
        return self._activation(X @ self.w + self.b)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Copmute the loss betweeen the current model predictions and true targets.

        Copmutes the MSE (Mean Squared Error) loss between the model predictions 
        on the input data X and the true targets y. 

        Parameters
        ----------
        X : ndarray
            Array of input feature data.
        y : ndarray
            Arrya of input target data. 

        Returns
        -------
        float
            Loss score between predicted and true targets. 
        """
        y_pred = self.predict(X)
        loss = self.compute_loss(y, y_pred)
        return loss

    def gradient_descent(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Performs parameter updates through the GD (Gradient Descent) algorithm.

        Performs both a forward pass by computing the predictions with the 
        current model values and then backward propegation by updating the 
        model parameters (coefficients and intercept) by the loss function 
        gradient w.r.t. said parameters. Contains logic for both GD,
        SGD (Stochastic Gradient Descent) and Mini-Batch SGD, which is accounted
        for by adjusting the batch size. The loss function and thus gradient updates
        are affected by the penalty term. 

        >>> loss(X, y) = mse(y, X^T @ W + b) + alpha * penalty
        
        Penalty given by:
        >>> l1_norm = sum(|w|)  # LASSO
        >>> l2_norm = sum(w^2)  # Ridge
        >>> l1_ratio * l1_norm + (1 - l1_ratio) * l2_norm  # Elastic Net
        
        'alpha' is the regularisation strength, set as a hyper parameter during 
        model instanciation.

        Parameters
        ----------
        X : ndarray
            Array of input feature training data.
        y : ndarray
            Array of input target training data. 

        Returns
        -------
        float
            The training loss between the predicted and true training targets. 
        """
        # One Epoch of training
        # Update parameters from mean loss of entire dataset
        # Using loops and slicing. NOTE Kept this to compare time complexity. 
        n_samples = X.shape[0]
        X, y = shuffle_data(X, y)

        n_batches = n_samples // self.batch_size
        n_batches += 1 if self.keep_rest else 0
        total_loss = 0

        for i in range(0, n_samples, self.batch_size):
            x_batch = X[i:i+self.batch_size]  # [n_batches, batch_size, n_features]
            y_batch = y[i:i+self.batch_size]  # [n_batches, batch_size]

            # exclude rest batch
            if not self.keep_rest and x_batch.shape[0] < self.batch_size:
                break  # break for the loop lap the rest_batch is being processed

            y_pred = self.predict(x_batch)

            # compute loss
            loss = self.compute_loss(y_batch, y_pred)
            total_loss += loss

            # gradient update (backward, step)
            self.gradient_update(x_batch, y_batch, y_pred)

        return total_loss / n_batches

    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient_update(self, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        pass

    @property
    def is_fitted(self) -> bool:
        """
        ...
        """
        return self.w is not None and self.b is not None

    @property
    def coefficients(self) -> np.ndarray:
        """
        Alias for weights (w). 
        """
        return self.w

    @property
    def intercept(self) -> float:
        """
        Alias for bias (b).
        """
        return self.b



# --------------------- LINEAR MODEL HELPER FUNCTIONS ---------------------

def lasso(coefficients : np.ndarray) -> float:
    """
    Least absolute shrinkage and selection operator (LASSO).

    Linear Regression and Logisitc Regression alias for l1 norm.

    Parameters
    ----------
    coefficients : ndarray
        Weights of the model. Intercept (bias) should be excluded. 

    Returns
    -------
    float
        LASSO regularisation pentaly term.
    """
    return l1_norm(coefficients)

def ridge(coefficients : np.ndarray) -> float:
    """
    Ridge (Ridge regression/-regularisation)
    
    Linear Regression and Logisitc Regression alias for l2 norm.

    Parameters
    ----------
    coefficients : ndarray
        Weights of the model. Intercept (bias) should be excluded. 

    Returns
    -------
    float
        Ridge regularisation pentaly term.
    """
    return l2_norm(coefficients)


def elastic_net(
       coefficients : np.ndarray,
       l1_ratio : float = 0.5,
       ) -> float:
    """
    Weighted sum of LASSO and Ridge terms. 

    Parameters
    ----------
    coefficients : ndarray
        Weights of the model. Intercept (bias) should be excluded. 
    l1_ratio : float
        Percentage of penalty to consist of l1 norm (LASSO) term. Remaining
        (1 - l1_ratio) is l2 norm (ridge). Expexted to be in range [0.0, 1.0].

    Returns
    -------
    float
        Elastic Net regularisation pentaly term.
    """
    return l1_ratio * l1_norm(coefficients) + (1-l1_ratio) * l2_norm(coefficients)



