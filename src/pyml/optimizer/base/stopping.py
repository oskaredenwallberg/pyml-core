import numpy as np

class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best = float("inf")
        self.count = 0

    def check(self, loss: np.number) -> bool:
        if loss < self.best:
            self.best = loss
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience
        
    def reset(self) -> None:
        self.best = float('inf')
        self.count = 0
