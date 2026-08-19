import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from ..base.decision_tree import (
    BinaryTreeNodeMeta,
    BinaryTreeNode,
    BinaryTree,
    DecisionTreeModel,
)
from ...utils.math import mse_reduction, mae_reduction
from ...utils.metrics import mse_score, mae_score
from ...utils.math import get_gaussian_pdf


# --------------------- DECISION TREE REGRESSOR CLASSES ---------------------
# - DecisionTreeRegressorNode  (BinaryTreeNode)
# - DecisionTreeRegressor      (DecisionTreeModel)


def criterion_score_regressor(
        targets : np.ndarray,
        splits : tuple[np.ndarray, np.ndarray],
        criterion : Literal['mse', 'mae'] = 'mse',
        ) -> float:
    """
    ...
    """
    if criterion == 'mse':
        return mse_reduction(targets, splits)
    if criterion == 'mae':
        return mae_reduction(targets, splits)


class DecisionTreeRegressorNode(BinaryTreeNode):
    """
    ...
    """
    def __init__(self, 
            indicies : np.ndarray, 
            depth : int,
            feature : int | None = None,
            threshold : int | float | None = None,
            targets : np.ndarray | None = None,
            ):
        super().__init__(indicies, depth, feature, threshold)
        self.targets = targets
        if self.is_leaf:
            ... # NOTE How does a Decision Tree Regressor output the target from an input? 
            # Is it mean value from the targets? 
            self.mean = np.mean(targets)  # FIXME Look this up...
            self.std = np.std(targets)  # Used to find split that minimizes variance
            # self.probs ??  # Distribution of values
            self.gaussian_pdf = get_gaussian_pdf(self.mean, self.std)  # use to compute probs 
        else:
            self.mean = None
            self.std = None
            self.gaussian_pdf = None
            # self.probs = None  NOTE Even needed? 

    def make_leaf(self) -> BinaryTreeNodeMeta:
        """
        ...
        """
        assert self.child_l.is_leaf and self.child_r.is_leaf, 'Error: Both children must be leaves. '
        assert not self.is_leaf, 'Error: Node is already a leaf. '
        self.child_l: DecisionTreeRegressorNode
        self.child_r: DecisionTreeRegressorNode
        self.targets = np.r_[self.child_l.targets, self.child_r.targets]
        self.mean = np.mean(self.targets)
        self.std = np.std(self.targets)
        self.gaussian_pdf = get_gaussian_pdf(self.mean, self.std)
        node_meta = BinaryTreeNodeMeta(
            child_l=self.child_l,
            child_r=self.child_r,
            feature=self.feature,
            threshold=self.threshold,
        )
        self.child_l = None
        self.child_r = None
        self.feature = None
        self.threshold = None
        return node_meta

    def revert_leaf(self, node_meta: BinaryTreeNodeMeta) -> None:
        """
        ...
        """
        assert self.is_leaf, 'Error: Node is already a leaf. '
        self.targets = None
        self.mean = None
        self.std = None
        self.gaussian_pdf = None
        # self.probs = ??? NOTE May not be needed
        self.child_l = node_meta.child_l
        self.child_r = node_meta.child_r
        self.feature = node_meta.feature
        self.threshold = node_meta.threshold



class DecisionTreeRegressor(DecisionTreeModel):
    """
    ...
    """
    def __init__(
            self,
            criterion : Literal['mse', 'mae'] = 'mse',
            max_depth : int = None,
            min_samples_leaf : int = 1,
            # min_samples_split : int  = 2,
            max_features : int | float = None,
            min_impurity_decrease : float = 0.0,
            random_state : int = None
        ):
        super().__init__(max_depth, min_samples_leaf, max_features, min_impurity_decrease, random_state)
        self.criterion = criterion
        # XXX Any corresponding self.n_classes for Regressor? 
        self.n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        ...
        """
        self.n_features = X.shape[1]
        n_samples = X.shape[0]

        feature, threshold = self._search_split(X, y)
        indicies = np.arange(0, n_samples)
        root = DecisionTreeRegressorNode(indicies=indicies, depth=0, feature=feature, threshold=threshold)
        self._tree = BinaryTree(root)
        queue = [self._tree.root]

        # perform splits until satisfactory seperation
        while queue:
            node: DecisionTreeRegressorNode = queue.pop(0)

            feature_values = X[node.indicies, node.feature]
            split_mask = feature_values < node.threshold
            split_indicies1 = node.indicies[split_mask]
            split_indicies2 = node.indicies[~split_mask]

            x1, x2 = X[split_indicies1], X[split_indicies2]
            y1, y2 = y[split_indicies1], y[split_indicies2]
            feature1, threshold1 = self._search_split(x1, y1)
            feature2, threshold2 = self._search_split(x2, y2)

            if self.max_depth is not None and node.depth == self.max_depth-1:
                feature1 = threshold1 = None
                feature2 = threshold2 = None
            
            node1 = DecisionTreeRegressorNode(
                indicies = split_indicies1,
                depth = node.depth+1,
                feature = feature1,
                threshold = threshold1,
                targets = y1 if feature1 is None else None
            )
            if feature1 is not None:  # found split, add Node
                queue.append(node1)
            
            node2 = DecisionTreeRegressorNode(
                indicies = split_indicies2,
                depth = node.depth+1,
                feature = feature2,
                threshold = threshold2,
                targets = y2 if feature2 is None else None
            )
            if feature2 is not None:
                queue.append(node2)

            node.add_children(left=node1, right=node2)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ...
        """
        n_samples = X.shape[0]
        predicions = np.zeros(n_samples, dtype=float)

        for i in range(n_samples):
            node: DecisionTreeRegressorNode = self._tree.root
            x = X[i]
            while not node.is_leaf:
                node = node.child_l if x[node.feature] < node.threshold else node.child_r
            predicions[i] = node.mean  # target predicion
        return predicions

    def reduced_error_pruning(self, X_val, y_val) -> None:
        """
        ...
        """
        if not self.is_fitted:
            raise AttributeError("Requires trained tree. Call .fit() first. ")
        
        # NOTE Could also be mae... Which one to prefer? 
        mse_best = mse_score(y_val, self.predict(X_val))

        # DFS Search
        # Iteratively remove leaves from the tree until no more can be removed without lowering mse_score on val data
        pruning = True
        while pruning:
            pruning = False
            stack = []
            stack.append(self._tree.root)

            while stack:
                node: DecisionTreeRegressorNode = stack.pop(0)

                # Only prune if both children are leaves
                if node.child_l.is_leaf and node.child_r.is_leaf:
                    node_meta = node.make_leaf()  # convert to leaf (prune children)

                    y_prune = self.predict(X_val)
                    mse_prune = mse_score(y_val, y_prune)

                    if mse_score < mse_best:
                        mse_best = mse_score
                        pruning = True
                    else:
                        node.revert_leaf(node_meta)
                    
                if not node.child_r.is_leaf:
                    stack.insert(0, node.child_r)
                if not node.child_l.is_leaf:
                    stack.insert(0, node.child_l)  # keep left most child on top of stack

    def _search_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, int|float]:
        """
        ...
        """
        split_feature = None
        split_threshold = None
        # FIXME Change later to match max_features
        best_score = self.min_impurity_decrease
        
        if self.max_features is not None:
            max_features = self.max_features if isinstance(self.max_features, int) else int(self.max_features*self.n_features)
            max_features = max(1, max_features)  # Ensure at least one feaure
            feature_indicies = self.rng.permutation(self.n_features)[:max_features]
        else:
            feature_indicies = np.arange(self.n_features)

        for i in feature_indicies:
            # No classes to partition by, thus splits will be made a set 20 number of times
            feature_values = x[:, i]
            ascending_values = np.sort(feature_values)
            thresholds = (ascending_values[1:] + ascending_values[:-1]) / 2  # find midpoints
            if len(thresholds) > 20:
                thresholds = np.linspace(np.min(thresholds), np.max(thresholds), num=21)

            for threshold in thresholds:
                split_mask = feature_values < threshold
                y1 = y[split_mask]  # left
                y2 = y[~split_mask]  # right

                # Each split must have at least min_samples_leafs in them
                if len(y1) < self.min_samples_leaf or len(y2) < self.min_samples_leaf:
                    continue
            
                split_score = criterion_score_regressor(targets=y, splits=(y1, y2), criterion=self.criterion)
                if split_score < best_score:  # '<' since we want to minimize 'mse' or 'mae'
                    best_score = split_score
                    split_feature = i
                    split_threshold = threshold

        return split_feature, split_threshold
