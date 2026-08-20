import numpy as np
from typing import Literal, NamedTuple
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# --------------------- BASE CLASSES ---------------------
# - BinaryTreeNodeMeta  (NamedTuple)
# - BinaryTreeNode      (ABC)
# - BinaryTree
# - DecisionTreeModel   (ABC)

class BinaryTreeNodeMeta(NamedTuple):
    """
    ...
    """
    child_l : 'BinaryTreeNode'
    child_r : 'BinaryTreeNode'
    feature : int
    threshold : int | float


class BinaryTreeNode(ABC):
    """
    ...
    """
    def __init__(self, 
            indicies : np.ndarray, 
            depth : int,
            feature : int | None = None,
            threshold : int | float | None = None,
            ):
        self.indicies = indicies
        self.depth = depth
        self.feature = feature  # index of feature
        self.threshold = threshold  # feature threshold where split is made
        self.child_l : BinaryTreeNode | None = None
        self.child_r : BinaryTreeNode | None = None

    def __str__(self) -> str:
        indicies_print = str(self.indicies[:5])[:-1] + '...' + str(self.indicies[-5:])[1:] \
            if self.indicies.shape[0] > 10 else str(self.indicies)
        if self.is_leaf:
            return f'L{self.depth}: indicies {indicies_print}'
        else:
            return f'N{self.depth}: indicies {indicies_print}, feature {self.feature}, threshold {self.threshold}'

    def add_children(self, 
            left : 'BinaryTreeNode | None' = None, 
            right : 'BinaryTreeNode | None' = None
        ) -> None:
        """
        Set the children of a decision node.

        The children are only updated if the input is of type `BinaryTreeNode`.
        Inputting None will not result in the current child being set to None but is there
        to allow one child being set at a time. The children, however, can either be leaf nodes, decision
        nodes or a mix of the two. 

        Parameters
        ----------
        left : BinaryTreeNode, default = None
            The left child of the decision node.
        right : BinaryTreeNode, default = None
            The right child of the decision node.

        Returns
        -------
        None
            This function does not return any value. 

        Exampels
        --------
        >>> node = BinaryTreeNode(...)  # must be decision node
        >>> left = BinaryTreeNode(...)  # either decision or leaf node
        >>> right = BinaryTreeNode(...)  # either decision or leaf node
        >>> node.add_children(left, right)  # Here both are added at the same time
        None
        """
        if isinstance(left, BinaryTreeNode):
            self.child_l = left
        if isinstance(right, BinaryTreeNode):
            self.child_r = right
    
    @property
    def is_leaf(self) -> bool:
        """
        A boolean value for whether this node is a leaf node or not. If not,
        it is by default a decision node. 

        Returns
        -------
        bool
            Boolean for whether the node is a leaf or not.
        """
        return self.child_l is None and self.child_r is None

    @abstractmethod
    def make_leaf(self) -> BinaryTreeNodeMeta:
        pass

    @abstractmethod
    def revert_leaf(self, node_meta: BinaryTreeNodeMeta) -> None:
        pass



class BinaryTree:
    """
    Breif descriptions here

    ...longer description here

    Parameters
    ----------
    root : BinaryTreeNode

    Properties
    ----------
    depth : int
        The maximum depth between the root node and most descendent leaf node.
    decision_nodes : list[BinaryTreeNode]
        A list of all decision nodes in the tree. These are the internal
        nodes used to split the data.
    leaf_nodes : list[BinaryTreeNode]
        A list of all leaf nodes in the the tree. These are the terminal
        nodes used to classify datapoints. 

    Examples
    --------
    >>> root = BinaryTreeNode(...)
    >>> tree = BinaryTree(root)
    """
    def __init__(self, root : BinaryTreeNode):
        self.root = root

    def __str__(self) -> str:
        return self.stringify()
    
    def stringify(self, order: Literal['dfs', 'bfs'] = 'dfs') -> str:
        """
        A string representing the binary tree from a depth first search traversal.

        Parameters
        ----------
        order : {'dfs', 'bfs'}, default = 'dfs'
            * dfs : Depth first search traversal when raveling tree to list
            * bfs : Bredth first search traversal when raveling tree to list

        Returns
        -------
        str
            The string print of the tree
        """
        nodes = self.ravel_dfs() if order == 'dfs' else self.ravel_bfs()
        return '\n'.join([f'{" |  "*node.depth}{node.__str__()}' for node in nodes])
    
    def ravel_dfs(self) -> list[BinaryTreeNode]:
        """
        Ravel the tree into a list of nodes through a depth first search (DFS).
        DFS inserts the children of the current node to the top of a stack, resulting 
        in the children of the node being priotised from left to right. 

        Returns
        -------
        list[BinaryTreeNodes]
            A list of the tree nodes ordered by a DFS.
        """
        nodes = []
        stack = [self.root]
        while stack:
            node : BinaryTreeNode = stack.pop(0)
            nodes.append(node)
            if not node.is_leaf:
                stack.insert(0, node.child_r)
                stack.insert(0, node.child_l)  # Keep left most on top
        return nodes

    def ravel_bfs(self) -> list[BinaryTreeNode]:
        """
        Flatten the tree into a list of nodes through a breadth first search (BFS). 
        BFS appends the children of a node to the back of a queue, resulting in the
        current level of nodes being prioritised from left to right. 

        Returns
        -------
        list[BinaryTreeNode]
            A list of the tree nodes ordered by a BFS.
        """
        nodes = []
        queue = [self.root]
        while queue:
            node : BinaryTreeNode = queue.pop(0)
            nodes.append(node)
            if not node.is_leaf:
                queue.append(node.child_l)
                queue.append(node.child_r)
        return nodes
    
    @property
    def depth(self) -> int:
        """
        The maximum depth of the tree. It is the number of nodes passed from
        the root of the tree to the leaf node furtherst away. The root node is
        at depth 0, meaning each new split yields an increment in depth. 

        Returns
        -------
        int
            The maximum depth of the tree.
        """
        return max(self.ravel_dfs(), key=lambda node: node.depth).depth

    @property
    def split_nodes(self) -> list[BinaryTreeNode]:
        """
        A list containing all decision nodes (no leaves) of a trained
        Decision Tree. This assumes that `fit()` has been called beforehand, 
        since no tree would be initialized otherwise. 
        
        Returns
        -------
        list[BinaryTreeNode]
            A list of all decision nodes in the tree.
        """
        return [node for node in self.ravel_dfs() if not node.is_leaf]
    
    @property
    def leaf_nodes(self) -> list[BinaryTreeNode]:
        """
        A list containing all leaf nodes of a trained Decision Tree. 
        This assumes that `fit()` has been called beforehand, since no tree 
        would be initialized otherwise. 

        Returns
        -------
        list[BinaryTreeNode]
            A list of all leaf nodes in the tree.
        """
        return [node for node in self.ravel_dfs() if node.is_leaf]
    

class DecisionTreeModel(ABC):
    def __init__(
            self,
            # criterion : Literal['mse', 'mae'] = 'mse',
            # criterion : Literal['info', 'gini'] = 'info',
            max_depth : int | None = None,
            min_samples_leaf : int = 1,
            # min_samples_split : int  = 2,
            max_features : int | float | None = None,
            min_impurity_decrease : float = 0.0,
            random_state : int | None = None
        ):
        # self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        # self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.rng = np.random.RandomState(random_state)

        self._tree: BinaryTree | None = None
        self.n_features = None
        # self.n_classes = None CLF

    def __str__(self) -> str:
        return self._tree.__str__()
    
    @property
    def is_fitted(self) -> bool:
        """
        A boolean value for whether the Decision Tree Classifier 
        has been fitted and has a _tree initialized or not. 
        True if the model is fitted, otherwise false. 

        Returns
        -------
        bool
            A boolean for whether the tree is fitted.
        """
        return self._tree is not None
    
    @property
    def depth(self) -> int:
        """
        ...
        """
        return self._tree.depth
    
    @property
    def n_splits(self) -> int:
        """
        ...
        """
        return len(self._tree.split_nodes)
    
    @property
    def n_leaves(self) -> int:
        """
        ...
        """
        return len(self._tree.leaf_nodes)
    
    @abstractmethod
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reduced_error_pruning(self, X_val:np.ndarray, y_val:np.ndarray) -> None:
        pass

    @abstractmethod
    def _search_split(self, x:np.ndarray, y:np.ndarray) -> tuple[int, int|float]:
        pass


# FIXME Make general function to plot both 
# - DecisionTreeClassifier splits
# - DecisionTreeRegressor splits
# NOTE Similar logic for both, can plot points for both types of models.
def plot_splits(
        model : DecisionTreeModel,
        X_train : np.ndarray,
        y_train : np.ndarray = None,
        feature_x : int = 0,
        feature_y : int = 1,
        ) -> None:
    """ 
    Plot the decision boundaries and classification fields the trained model has 
    learned from the training data. 

    The vertical and horisontal black lines represent the split thresholds of the 
    decision nodes and the colored fields inbetween the lines represent leaf 
    classification areas. A vertical line represents a split along feature x and 
    a horisontal line represents a split along feature y. This function iteratates
    through all decision nodes of the tree: (i) extracts the split threshold as an 
    offset for the line, (ii) begins to draw the line starting from the last decision 
    boundary until it meets the first crossing decision 
    boundary or the end of the plot. Each line drawn is stored in a tuple of 
    (offset, line_start, line_stop) as a reference for future lines. For each leaf, 
    the field surrounding the datapoints of the leaf is colored according to
    the majority label and represents a classification area for future datapoints. 
    The training datapoints are then represented with a scatter, also color coded
    according to their label.

    Parameters
    ----------
    model : DecisionTreeClassifier
        A trained Decision Tree Classifier model. The model is assumed to have been
        trained on X_train and y_train using the `fit()` method before plotting.
    X_train : ndarray
        The feature data used to train the model. X_train is assumed to be a 2D
        array and to only contain numerical values of the same scale as when the model
        was fitted. 
    y_train : ndarray
        The class labels used to train the model. y_train is assumed to be a 1D
        array and to contain integers representing the different labels classes.
    feature_x : int
        The index of the feature to be used as the x axis in the plot.
    feature_y : int
        The index of the feature to be used as the y axis for the plot.
    
    Returns
    -------
    None
        This function does not return any values. Instead it directly plots the decision
        boundaries. 

    Raises
    ------
    ValueError
        ...
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([  # Example training data
            [2.1, 7.3, 0.8, 5.9, 2.0],  # index 0
            [4.6, 6.7, 1.2, 0.6, 8.1],  # index 1
            [2.3, 4.9, 2.7, 8.5, 8.8],  # index 2
        ]).T
    >>> y = np.array([0, 1, 0, 1, 1])
    >>> dt_clf = DecisionTreeClassifier()
    >>> dt_clf.fit(X, y)
    >>> plot_splits(dt_clf, X, y, 0, 1)  # Splits seen from features indexed by 0 & 1
    None
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap('tab10')
    
    xmin, xmax = X_train[:,feature_x].min()-1, X_train[:,feature_x].max()+1
    ymin, ymax = X_train[:,feature_y].min()-1, X_train[:,feature_y].max()+1
    # (line_offset, line_start, line_stop), e.g. (x, y1, y2) or (y, x1, x2)
    vlines = [(xmin, ymin, ymax), (xmax, ymin, ymax)]  #   vertical lines (x, y1, y2)
    hlines = [(ymin, xmin, xmax), (ymax, xmin, xmax)]  # horizontal lines (y, x1, x2)

    for node in model._tree.split_nodes:
        node: BinaryTreeNode
        y_subset:np.ndarray = X_train[node.indicies, feature_y]
        x_subset:np.ndarray = X_train[node.indicies, feature_x]

        # Draw decision boundaries (splits) where datapoints are bisected for children.
        if node.feature == feature_x:
            # Filter out horisontal lines that do not overlap the x threshold
            hlines_filtered = [line for line in hlines if line[1] < node.threshold < line[2]]
            # Find thresholds closest to outer edges of feature subsets used for splits
            ymin = max(line[0] for line in hlines_filtered if line[0] < y_subset.min())
            ymax = min(line[0] for line in hlines_filtered if line[0] > y_subset.max())
            vlines.append((node.threshold, ymin, ymax))  # store line plotted
            ax.vlines(x=node.threshold, ymin=ymin, ymax=ymax, colors='k')
        if node.feature == feature_y:
            # Filter out vertical lines that do not overlap the y threshold
            vlines_filtered = [line for line in vlines if line[1] < node.threshold < line[2]]
            # Find thresholds closest to outer edges of feature subsets used for splits
            xmin = max(line[0] for line in vlines_filtered if line[0] < x_subset.min())
            xmax = min(line[0] for line in vlines_filtered if line[0] > x_subset.max())
            hlines.append((node.threshold, xmin, xmax))  # store line plotted
            ax.hlines(y=node.threshold, xmin=xmin, xmax=xmax, colors='k')

        # Highlight leaf areas where datapoints are classified (colored by label)
        for child in [node.child_l, node.child_r]:
            if not child.is_leaf:
                continue
            x_subset = X_train[child.indicies, feature_x]
            y_subset = X_train[child.indicies, feature_y]
            xmin, xmax = x_subset.min(), x_subset.max()
            ymin, ymax = y_subset.min(), y_subset.max()
            # Filter out lines not overlapping the leaf's datapoints
            vlines_filtered = [line for line in vlines if line[1] < ymin and ymax < line[2]]
            hlines_filtered = [line for line in hlines if line[1] < xmin and xmax < line[2]]
            # We now seek the decison boundaries closest to the leaf's datapoints (using offset)
            xmin = max([line[0] for line in vlines_filtered if line[0] < xmin])
            xmax = min([line[0] for line in vlines_filtered if line[0] > xmax])
            ymin = max([line[0] for line in hlines_filtered if line[0] < ymin])
            ymax = min([line[0] for line in hlines_filtered if line[0] > ymax])

            ax.fill([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax], c=cmap(child.label), alpha=0.2)

    offsets_x = [line[0] for line in vlines]
    offsets_y = [line[0] for line in hlines]

    colors = [cmap(l) for l in y_train] if y_train is not None else 'k'
    ax.scatter(x=X_train[:,feature_x], y=X_train[:,feature_y], c=colors)
    ax.set_xlim(min(offsets_x), max(offsets_x))
    ax.set_ylim(min(offsets_y), max(offsets_y))
    ax.set_xlabel(f'Feature {feature_x}')
    ax.set_ylabel(f'Feature {feature_y}')
    plt.show()




# Example tree
# Here the first N2 could have its leaf children (L3, L3) being pruned!
# Though the second N2 could not have its children (L3, N3) pruned.
# N0: indicies [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16], feature 1, threshold 2 
#  |  N1: indicies [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13], feature 1, threshold 1 
#  |   |  N2: indicies [ 8 11 12 13], feature 0, threshold 6 
#  |   |   |  L3: indicies [11 12] 
#  |   |   |  L3: indicies [ 8 13] 
#  |   |  N2: indicies [ 0  1  2  3  4  5  6  7  9 10], feature 0, threshold 4 
#  |   |   |  L3: indicies [ 1  3 10] 
#  |   |   |  N3: indicies [0 2 4 5 6 7 9], feature 0, threshold 56 
#  |   |   |   |  L4: indicies [0 4 5 6 7] 
#  |   |   |   |  L4: indicies [2 9] 
#  |  L1: indicies [14 15 16]

