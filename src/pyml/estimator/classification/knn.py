import numpy as np

from ..base.neighbor import KNNModel

# --------------------- K NEAREST NEIGHBORS CLASSIFIER ---------------------
# - KNNClassifier       (KNNModel)

class KNNClassifier(KNNModel):
    def __init__(
            self, 
            k = 5
            ):
        super().__init__(k)
    
    def evaluate(self, distances) -> np.ndarray:
        """
        Compute the highest probability classification through majority voting.

        Parameters
        ----------
        distances : ndarray
            An array of the distances between the training datapoints and the 
            one to predict. 

        Returns
        -------
        ndarray
            An array of a single value, the predicted label. The majority vote.
        """
        k_indicies = np.argsort(distances)[:self.k]  # indicies of k nearest points
        k_labels = self.y_train[k_indicies]
        k_distances = distances[k_indicies]

        # labels and label counts of k nearest, and mean distance of counted points
        votes, vote_counts = np.unique(k_labels, return_counts=True)
        mean_distances = np.array([k_distances[k_labels == l].mean() for l in votes])

        # index of the label with the majority count, may be more than one if tied count
        majority = np.argwhere(vote_counts == vote_counts.max()).flatten()

        # tiebreaker - choose label index with shortest mean distance
        majority = majority[np.argmin(mean_distances[majority])]

        return votes[majority]
