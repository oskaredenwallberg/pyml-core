import numpy as np
from typing import Literal

from ..base import (
    BinaryTreeNodeMeta,
    BinaryTreeNode,
    BinaryTree,
    DecisionTreeModel,
)

from ...utils.math import info_gain, gini_gain
from ...utils.metrics import accuracy_score


# --------------------- DECISION TREE CLASSIFIER CLASSES ---------------------
# - DecisionTreeClassifierNode  (BinaryTreeNode)
# - DecisionTreeClassifier      (DecisionTreeModel)


def criterion_score_classifier(
    labels : np.ndarray, 
    splits : tuple[np.ndarray, np.ndarray], 
    criterion : Literal['info','gini'] = 'info'
    ) -> float:
    """ ### Compute Criterion Score

    Top level API for criterion score selection.

    See specific criterion functions for details regarding how each score is computed.
    See `info_gain()` for info criterion and `gini_gain()` for gini criterion.
    This function is meant to be the top level api used in DecisionTreeClassifier class. 

    Params
    ------
    labels : ndarray
        An array of original dataset labels. 
    splits : tuple[ndarray, ndarray]
        A tuple of arrays the original dataset was split into.
    criterion : {'info', 'gini'}, default = 'info'
        The criterion to use for the gain from performing the split.
        - 'info' : Information Gain (Entropy reduction)
        - 'gini' : Gini Gain (Impurity reduction)

    Returns
    -------
    float
        Impurity Reduction by performing split.
    """
    if criterion == 'info':
        return info_gain(labels=labels, splits=splits)
    if criterion == 'gini':
        return gini_gain(labels=labels, splits=splits)

class DecisionTreeClassifierNode(BinaryTreeNode):
    """
    ...

    Parameters
    ----------
    indicies : ndarray
        ...
    depth : int
        ...
    feature : str, default = None
        ...
    threhold : int|float, default = None
        ...
    labels : ndarray, default = None
        ...

    Attributes
    ----------
    child_l : DecisionTreeNode, default = None
        ...
    child_r : DecisionTreeNode, default = None
        ...

    Properties
    ----------
    is_leaf : bool
        ...
    
    Examples
    --------
    >>> # Examples of decision node

    >>> # Exampels of leaf node
    """
    def __init__(self, 
            indicies : np.ndarray,
            depth : int,
            feature : int | None = None,
            threshold : int | float | None = None,
            labels : np.ndarray | None = None,
            ):
        super().__init__(indicies, depth, feature, threshold)
        self.labels = labels
        if self.is_leaf:
            unique, counts = np.unique(self.labels, return_counts=True)
            self.label = unique[np.argmax(counts)]  # takes first instance if shared for max
            self.probs = counts / np.sum(counts)
        else:  # is node
            self.label = None
            self.probs = None

    def make_leaf(self) -> BinaryTreeNodeMeta:
        """
        asdasd

        ...

        Returns
        -------
        dict
            ...
        """
        assert self.child_l.is_leaf and self.child_r.is_leaf, 'Error: Both children must be leaves. '
        assert not self.is_leaf, 'Error: Node is already a leaf. '
        self.child_l : DecisionTreeClassifierNode
        self.child_r : DecisionTreeClassifierNode
        self.labels = np.r_[self.child_l.labels, self.child_r.labels]  # take childern labels
        unique, counts = np.unique(self.labels, return_counts=True)
        self.label = unique[np.argmax(counts)]  # takes first instance if shared for max
        self.probs = counts / np.sum(counts)
        node_meta = BinaryTreeNodeMeta(
            child_l=self.child_l,
            child_r=self.child_r,
            feature=self.feature,
            threshold=self.threshold
        )
        self.child_l = None
        self.child_r = None
        self.feature = None
        self.threshold = None
        return node_meta

    def revert_leaf(self, node_meta: BinaryTreeNodeMeta) -> None:
        """
        asdad

        ...
        
        Parameters
        ----------
        node_meta : dict
            ...

        Returns
        -------
        None
            ...
        
        """
        assert self.is_leaf, 'Error: Node is already a leaf. '
        self.labels = None
        self.label = None
        self.probs = None
        self.child_l = node_meta.child_l
        self.child_r = node_meta.child_r
        self.feature = node_meta.feature
        self.threshold = node_meta.threshold


# Attributes (features) are assumed to be:
#   * Info: Categorical
#   * Gini: Continuous
class DecisionTreeClassifier(DecisionTreeModel):
    """
    Decision trees for classification.

    A Decision Trees algorithm used for classification tasks. It iteratively
    bisects (splits) the data based on the features with the goal to reach pure 
    subgroups of class labels. Each internal node represents a decision point
    and each leaf (terminal node) represents a subgroup of labels used to 
    classify new datapoints through majority voting. 

    Parameters
    ----------
    criterion : {'info', 'gini'}, default = 'info'
        The type of measure used for assessing the quality of the split.
    max_depth : int, default = None
        The maximum depth of the tree. I.e. the maximum levels of nodes (incl. 
        leaves) the Decision Tree can reach during training. When set to None
        the tree can grow unlimited. Used to limit model complexity.
    min_samples_leaf : int, default = 1
        The minimum required number of samples needed to form a leaf. Used 
        to avoid overfitting.
    min_sample_split : int, default = 2
        ...
    max_features : int | float, default = None
        The maximum number of features to be considered when computing
        the best split. Features are randomly selected. Integer values
        specifies the number of features, and floats the fraction of all 
        features to be considered. 
    min_gain : float, default = 0.0
        The minimum criterion score required for a split to be considered.
    random_state : int, default = None
        Any integer value used to create a RandomState object. Used for 
        reproducibility.

    Attributes
    ----------
    _tree : BinaryTree
        The tree object containing the nodes and leaves. This attribute is
        only initialized after `fit()` has been called. 
    n_classes : int
        The number of classes present in the label training data. This attribute
        is only initialized after `fit()` has been called. 
    n_features : int
        The number of features present in the feature training data. This 
        attribute is only initialized after `fit()` has been called. Both
        validation and testing data must contain the same features, in the
        same order as the training data. 

    Properties
    ----------
    is_fitted : bool
        A boolean for whether the tree is fitted or not
    """
    def __init__(
            self, 
            criterion : Literal['info', 'gini'] = 'info',
            max_depth : int = None,
            min_samples_leaf : int = 1,
            # min_samples_split : int = 2,  # !FIXME! 
            max_features : int | float = None,
            min_impurity_decrease : float = 0.0,
            random_state : int = None,
        ):
        super().__init__(max_depth, min_samples_leaf, max_features, min_impurity_decrease, random_state)
        self.criterion = criterion
        self.n_classes = None
        self.n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Decision Tree to the training data inputs.

        ... longer desciption

        Parameters
        ----------
        X : ndarray
            An array of feature training data. X is assumed to be a 2D array
            of numerical values.
        y : ndarray
            An array of label class data. y is assumed to be a 1D array of
            integer values representing the type of class. 

        Returns
        -------
        None
            This method does not reutrn any value. It initializes and builds a
            decision tree in place.

        Raises
        ------
        ValueError
            ...

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([  # Example training data
                [2.1, 7.3, 0.8, 5.9, 2.0],  # feature 0
                [4.6, 6.7, 1.2, 0.6, 8.1],  # feature 1
                [2.3, 4.9, 2.7, 8.5, 8.8],  # feature 2
            ]).T
        >>> y = np.array([0, 1, 0, 1, 1])
        >>> dt_clf = DecisionTreeClassifier()
        >>> dt_clf.fit(X, y)
        None
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        n_samples = X.shape[0]

        feature, threshold = self._search_split(X, y)
        indicies = np.arange(0, n_samples)
        root = DecisionTreeClassifierNode(indicies=indicies, feature=feature, threshold=threshold, depth=0)  # depth 0
        self._tree = BinaryTree(root)
        queue = [self._tree.root]

        # perform splits until perfectly separated classes or stop before min_samples_leaf is reached
        while queue:
            node: DecisionTreeClassifierNode = queue.pop(0)

            feature_values = X[node.indicies, node.feature]
            split_mask = feature_values < node.threshold
            split_indicies1 = node.indicies[split_mask]
            split_indicies2 = node.indicies[~split_mask]

            x1, x2 = X[split_indicies1], X[split_indicies2]
            y1, y2 = y[split_indicies1], y[split_indicies2]
            feature1, threshold1 = self._search_split(x1, y1)
            feature2, threshold2 = self._search_split(x2, y2)

            # Assert max depth for tree
            if self.max_depth is not None and node.depth == self.max_depth-1:
                feature1 = threshold1 = None  # Will cause splits to become leafs
                feature2 = threshold2 = None

            node1 = DecisionTreeClassifierNode(
                indicies=split_indicies1,
                depth=node.depth+1,
                feature=feature1,       # None for leaf
                threshold=threshold1,   # None for leaf
                labels=y1 if feature1 is None else None  # None for node
            )
            if feature1 is not None:    # found split, add Node
                queue.append(node1)

            node2 = DecisionTreeClassifierNode(
                indicies=split_indicies2,
                depth=node.depth+1,
                feature=feature2,       # None for leaf
                threshold=threshold2,   # None for leaf
                labels=y2 if feature2 is None else None  # None for node
            )
            if feature2 is not None:    # found split, add Node
                queue.append(node2)
                
            node.add_children(left=node1, right=node2)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classify the labels for the input data X.

        Each datapoint is classified by walking through the tree until arriving
        at a leaf node carrying the label. Starting at the root node, the algorithm 
        moves to the next 
        node by comparing a feature value of the datapoint with the threshold of
        the current decision node. Eventually it arrives at a leaf node, carrying the 
        label used to predict the class of that datapoint. This algorithm iterates 
        through the datapoints, classifying them one at a time.

        Parameters
        ----------
        X : ndarray
            Feature data containing datapoints to be classified.

        Returns
        -------
        ndarray
            An array of predicted labels for the datapoints.

        Raises
        ------
        ValueError
            ...
        
        Examples
        --------
        >>> # Illustrative example of how datapoints are classified:
        >>> # datapoint x, node n, leaf l
        >>> # x1 : root -> n1 -> n3 -> l1 (class=1)
        >>> # x2 : root -> n2 -> l2 (class=0)
        >>> # x3 : root -> n2 -> n6 -> l3 (class=1)
        >>> import numpy as np
        >>> X = np.array([
        >>>     [3.4 5.0],
        >>>     [1.2 0.9],
        >>>     [1.8 4.7]
        >>>     ])
        >>> predict(X)
        [1, 0, 1]
        """
        n_samples = X.shape[0]
        predictions = -np.ones(n_samples, dtype=int)  # minus ones as non predictions

        for i in range(n_samples):
            node: DecisionTreeClassifierNode = self._tree.root
            x = X[i]
            while not node.is_leaf:
                node = node.child_l if x[node.feature] < node.threshold else node.child_r
            predictions[i] = node.label
        return predictions
    
    def reduced_error_pruning(self, X_val:np.ndarray, y_val:np.ndarray) -> None:
        """
        Reduces the size of the Decision Tree by pruning leaves until accuracy 
        starts decaying. 

        Reduced Error Pruning is a pruning technique used to remove complexity from
        the model (Decision Tree) without harming the performance of it. The idea is
        to find the least complex model that still generalises well. This is a means to 
        reduce variance from an overfitted model. Reduced Error Pruning works through an 
        iterative process where each node with a pair of leaf children has its leaves 
        pruned and itself become converted into a leaf. The model is then evaluated on 
        validation data to measure how performance has changed. If (i) the accuracy of the 
        model is the same as or better than before, those two leaves are kept pruned. 
        Else if (ii) the accuracy of the model decreases, the pruning is reverted and the 
        node restored. When all pairs of leaves have been tested for pruning, without 
        success in improving the accuracy against the validation data, the process stops.

        Parameters
        ----------
        X_val : ndarray
            The validation feature data used to make new predictions on unseen data. X_val 
            is assumed to be a 2D array containing the same features as the data the 
            Decision Tree was trained on.
        y_val : ndarray
            The validation class labels used to evaluate the accuracy of the predictions
            against. y_val is assumed to be a 1D array of integers representing the label
            classes. 

        Returns
        -------
        None
            This method does not return any value. Instead it modifies the 
            `DecisionTreeClassifier` object in place. 

        Raises
        ------
        ValueError
            ...

        Examples
        --------
        >>> dt_clf = DecisionTreeClassifier()
        >>> dt_clf.fit(X_train, y_train)
        >>> dt_clf.reduced_error_pruning(X_val, y_val)
        None

        Notes
        -----
        * Use `plot_splits()` or print the model using `print(dt_clf)` before and 
        after calling `reduced_error_pruning` to see how the model has changed.
        """
        # 1. trained tree assumed
        if not self.is_fitted:
            raise AttributeError("Requires trained tree. Call .fit() first. ")
        
        # 2. compute base accuracy on validation data
        accuracy_best = accuracy_score(y_val, self.predict(X_val))  # previous accuracy on last pruned version

        # 3. itertively remove leaves until no leaf can be remove without lowering accuracy on validation data
        # DFS search through tree, lower memory complexity than BFS
        pruning = True
        while pruning:
            pruning = False
            stack = []
            stack.append(self._tree.root)

            while stack:
                node: DecisionTreeClassifierNode = stack.pop(0)
                
                # Only prune if both children are leaves
                if node.child_l.is_leaf and node.child_r.is_leaf:
                    node_meta = node.make_leaf()  # Turn node into leaf (prune children)
                    
                    y_prune = self.predict(X_val)
                    accuracy_prune = accuracy_score(y_val, y_prune)

                    if accuracy_prune >= accuracy_best:  # New leaf is kept!
                        accuracy_best = accuracy_prune
                        pruning = True  # More pruning possible
                    else:  # Pruning made accuracy worse
                        node.revert_leaf(node_meta)  # Restore leaf back to node
                
                if not node.child_r.is_leaf:  # Removed: node.child_l is not None and 
                    stack.insert(0, node.child_r)
                if not node.child_l.is_leaf:  # Removed: node.child_r is not None and 
                    stack.insert(0, node.child_l)  # keep left most child always on top of stack

    def _search_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, int|float]:
        """
        Compute the feature and threshold that generates the best split according to the 
        criterion function.

        This is an internal helper function used for the `fit()` method. This function 
        loops over all features in data x and then for each such feature loops over 
        different thresholds (midpoints between two values) to perform a bisection of 
        the data into two groups with the goal of dividing the labels of y into the purest 
        possible subsets. The purity of this partitioning of labels is scored according to 
        a criterion measure: information gain (i.e. reduction of entropy) or gini impurity 
        (i.e. reduction of probability of misclassification)

        Parameters
        ----------
        x : ndarray
            Feature data that will decide next split. x is assumed
            to be a 2d array with all numerical values.
        y : ndarray
            Label data used to measure quality of a split. y is assumed 
            to be a 1d array with all integer values. 

        Returns
        -------
        tuple[int, int|float]
            A tuple containing: 
            * split_feature (int) Feature index yielding best split 
            * split_threshold (int|float) Optimal threshold to perform split at.

        Raises
        ------
        ValueError
            If x or y blah blah
        
        Examples
        --------
        >>> import numpy as np
        >>> x = np.array([
        >>>     [2.5, 3.0], 
        >>>     [1.2, 4.5], 
        >>>     [3.3, 2.1]
        >>>     ])
        >>> y = np.array([0, 1, 0])
        >>> _search_split(x, y)
        (1, 3.0)

        Notes
        -----
        - This is an internal function, not to be called by end users. 
        """
        split_feature = None
        split_threshold = None
        # init best split score to the minimum requirement, worst possible scenario
        best_score = self.min_impurity_decrease

        # Reduce features used for each split to maximum number or fraction specified
        if self.max_features is not None:
            max_features = self.max_features if isinstance(self.max_features, int) else int(self.max_features*self.n_features)
            max_features = max(1, max_features)  # Ensure at least one feaure
            feature_indicies = self.rng.permutation(self.n_features)[:max_features]
        else:
            feature_indicies = np.arange(self.n_features)
        
        for i in feature_indicies:
            feature_values = x[:, i]
            ascending_uniques = np.sort(np.unique(feature_values)) # exclude first
            thresholds = (ascending_uniques[1:] + ascending_uniques[:-1]) / 2  # find midpoints
            if len(thresholds) > 20:
                thresholds = np.linspace(start=np.min(feature_values), stop=np.max(feature_values), num=21) 

            for threshold in thresholds:
                split_mask = feature_values < threshold
                y1 = y[split_mask]  # left
                y2 = y[~split_mask]  # right
                
                # Each split must have at least min_samples_leafs in them
                if len(y1) < self.min_samples_leaf or len(y2) < self.min_samples_leaf:
                    continue
                
                split_score = criterion_score_classifier(labels=y, splits=(y1, y2), criterion=self.criterion)
                if split_score > best_score:  # '>' since we want ot max 'info gain' or 'gini gain'
                    best_score = split_score
                    split_feature = i
                    split_threshold = threshold
        
        return split_feature, split_threshold
