import numpy as np

class EarlyStopper:
    def __init__(self, patience: int = 5):
        pass
        self.patience = patience  # set -1 to ignore early stopping
        self.best_loss = float('inf')  # init to worst possible
        self.counter = 0

    def check_stop(self, loss: np.number) -> bool:
        """..."""
        if self.patience < 0:
            return False  # ignore early stopping
        if loss < self.best_loss:
            self.reset(loss)
        else:        
            self.counter += 1
        return self.counter >= self.patience
        
    def reset(self, loss: np.number = float('inf')) -> None:
        """..."""
        self.best_loss = loss
        self.counter = 0