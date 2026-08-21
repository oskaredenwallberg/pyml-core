import numpy as np
from typing import Literal

from ..base.linear import LinearModel
from ..base.linear import lasso, ridge, elastic_net
from ...utils.math.activations import sigmoid
from ...utils.metrics.classification import bce_score


class LogisiticLASSO(LinearModel):
    def __init__(
            self, 
            learning_rate = 0.01, 
            alpha : float = 1,  # regularisation strength
            epochs = 200,
            batch_size = 32,
            keep_rest = False,
            training_method = 'mbsgd',
            patience = 5,
            train_val_split = 0.2,
            verbose = True):
        super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
        self._activation = sigmoid
        self.alpha = alpha
        self.n_features = None

    def compute_loss(self, y_true, y_pred) -> float:
        """
        ...
        """
        penalty = lasso(self.w)  # l1 norm
        loss = bce_score(y_true, y_pred) + self.alpha * penalty
        return loss

    def gradient_update(self, x, y_true, y_pred) -> None:
        """
        >>> loss(X, y) = -1/m * sum( y * log(h) + (1-y) * log(1-h) ) + alpha * sum(|w|)
        >>> # LASSO partial differentials
        >>> d/dw loss(w, b) = 1/m sum( x * [h(x) - y] ) + alpha * sign(w)
        >>> d/db loss(w, b) = 1/m sum( h(x) - y )
        """
        error = y_pred - y_true
        m = x.shape[0]

        gradient_w = x.T @ error / m + self.alpha * np.sign(self.w)
        gradient_b = error.mean()

        self.w -= self.learning_rate * gradient_w
        self.b -= self.learning_rate * gradient_b


class LogisticRidge(LinearModel):
    def __init__(
            self, 
            learning_rate = 0.01, 
            lamda : float = 1,  # regularisation strength
            epochs = 200,
            batch_size = 32,
            keep_rest = False,
            training_method = 'mbsgd',
            patience = 5,
            train_val_split = 0.2,
            verbose = True):
        super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
        self._activation = sigmoid
        self.lamda = lamda
        self.n_features = None

    def compute_loss(self, y_true, y_pred) -> float:
        """
        ...
        """
        penalty = ridge(self.w)
        loss = bce_score(y_true, y_pred) + self.lamda * penalty
        return loss
    
    def gradient_update(self, x, y_true, y_pred) -> None:
        """
        >>> loss(X, y) = -1/m * sum( y * log(h) + (1-y) * log(1-h) ) + lamda * sum(w^2)
        >>> # Ridge partial differentials
        >>> d/dw loss(w, b) = 1/m sum( x * [h(x) - y] ) + 2 * lamda * sum(w)
        >>> d/db loss(w, b) = 1/m sum( h(x) - y )
        """
        error = y_pred - y_true
        m = x.shape[0]

        gradient_w = x.T @ error / m + self.lamda * self.w
        gradient_b = error.mean()

        self.w -= self.learning_rate * gradient_w
        self.b -= self.learning_rate * gradient_b



# class LogisticRegression(LinearModel):
#     def __init__(
#             self, 
#             learning_rate = 0.01, 
#             epochs = 200, 
#             batch_size = 32,
#             keep_rest = False,
#             training_method = 'mbsgd',
#             patience = 5,
#             train_val_split = 0.2,
#             verbose = True):
#         super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
#         self._activation = sigmoid
#         self.n_features = None

#     def compute_loss(self, y_true, y_pred):
#         """
#         Includes no penalty in normal LogisticRegression
#         """
#         loss = bce_score(y_true, y_pred)  # binary cross entropy score
#         return loss

#     def gradient_update(self, x, y_true, y_pred):
#         """
#         Includes no regularisation penalty termin in normal LogisticRegression. 

#         >>> loss(X, y) = -1/m * sum( y * log(h(X)) + (1-y) * log(1-h(X)) )  # nll
#         >>> where, h(x) = 1/(1+e^-(wx+b))
#         >>> # Logistic Regression partial differentials
#         >>> d/dw loss(w, b) = 1/m sum( x * [h(x) - y] )
#         >>> d/db loss(w, b) = 1/m sum( h(x) - y )
#         """ # NOTE see calculation at bottom of file
#         error = y_pred - y_true  # h(x) - y 
#         m = x.shape[0]

#         gradient_w = x.T @ error / m
#         gradient_b = error.mean()

#         self.w -= self.learning_rate * gradient_w
#         self.b -= self.learning_rate * gradient_b


# class LogisticElasticNet(LinearModel):
#     def __init__(
#             self, 
#             learning_rate = 0.01,
#             l1_ratio : float = 0.5, 
#             alpha : float = 1,  # regularisation strength
#             epochs = 200,
#             batch_size = 32,
#             keep_rest = False,
#             training_method = 'mbsgd',
#             patience = 5,
#             train_val_split = 0.2,
#             verbose = True):
#         super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
#         self._activation = sigmoid
#         self.alpha = alpha
#         self.l1_ratio = l1_ratio
#         self.n_features = None   

#     def compute_loss(self, y_true, y_pred) -> float:
#         """
#         ...
#         """
#         penalty = elastic_net(self.w)  # weighted sum of l1 norm and l2 norm
#         loss = bce_score(y_true, y_pred) + self.alpha * penalty
#         return loss
    
#     def gradient_update(self, x, y_true, y_pred) -> None:
#         """
#         >>> penalty(w) = l1_ratio * sum(|w|) + (1 - l1_ratio) * sum(w^2)
#         >>> loss(X, y) = -1/m * sum( y * log(h) + (1-y) * log(1-h) ) + lamda * sum(w^2)
#         >>> # Ridge partial differentials
#         >>> d/dw penalty(w) = l1_ratio * sign(w) + (1 - l1_ratio) * 2 * w
#         >>> d/dw loss(w, b) = 1/m sum( x * [h(x) - y] ) + alpha * d/dw penalty(w)
#         >>> d/db loss(w, b) = 1/m sum( h(x) - y )
#         """
#         m = x.shape[0]
#         errors = y_pred - y_true

#         gradient_penalty = self.l1_ratio * np.sign(self.w) + (1 - self.l1_ratio) * 2 * self.w
#         gradient_w = x.T @ errors / m + self.alpha * gradient_penalty
#         gradient_b = errors.mean()

#         self.w -= self.learning_rate * gradient_w
#         self.b -= self.learning_rate * gradient_b



#  Logistic Regression Partial Derivatives 
#  Calculations w/o penalty term. 
#  loss (a.k.a. NLL or BCE)
#  
#  loss = - sum( y * log(1/(1+e^-(wx+b))) +  (1-y) * log(1 - 1/(1+e^-(wx+b))) )
#  * { 
#    log(1 - 1/(1+e^-(wx+b))) =
#    = log((1+e^-(wx+b))/(1+e^-(wx+b)) - 1/(1+e^-(wx+b)))
#    = log( e^-(wx+b)/(1+e^-(wx+b)) )
#  }
#  loss = - sum( y * log(1/(1+e^-(wx+b))) + (1-y) * (log(e^-(wx+b)) - log((1+e^-(wx+b))) )
#  loss = - sum( -y*log(1+e^-(wx+b)) + y*(wx+b) + y*log((1+e^-(wx+b))) + -(wx+b) - log((1+e^-(wx+b)) )
#  loss = - sum(  y*(wx+b) -(wx+b) - log((1+e^-(wx+b)) )
#  loss = - sum(  (y-1)*(wx+b) - log((1+e^-(wx+b)) )
#  
#  d/dw loss = -1/m sum( x * [(y-1) - e^-(wx+b)/(1+e^-(wx+b))] )
#  d/db loss = -1/m sum( (y-1) - e^-(wx+b)/(1+e^-(wx+b)) )
#  
#  h(x) = 1 / (1 + e^-(wx+b))  <==>  e^-(wx+b) = 1/h(x) - 1
#  ==> e^-(wx+b) * 1/(1+e^-(wx+b)) = (1/h(x) - 1) * h(x) = 1 - h(x)
#  ==> 1 - y - (1 - h(x)) = h(x) - y
#  
#  d/dw loss = 1/m sum( x * [h(x) - y] )
#  d/db loss = 1/m sum( h(x) - y )




# L = p^y * (1-p)^(1-y)
# 
# LL = y*log(p) + (1-y)log(1-p)
# 
#              m
# df/db = -1/m*∑((yi-1) + e^-(w*xi+b)/(1+e^-(w*xi+b)))
#             i=1
# 
#              m
# df/dw = -1/m*∑xi((yi-1) + e^-(w*xi+b)/(1+e^-(w*xi+b)))
#             i=1

# rewrite the loss function and then take the derivative!