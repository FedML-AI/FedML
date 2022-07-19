# MIT License
#
# Copyright (C) IBM Corporation 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Random Forest Classifier with Differential Privacy
"""
from collections import defaultdict, namedtuple
import numbers
import warnings
from joblib import Parallel, delayed
import numpy as np

from sklearn.utils import check_array
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier as BaseDecisionTreeClassifier

from fedml.core.differential_privacy.accountant import BudgetAccountant
from fedml.core.differential_privacy.utils import PrivacyLeakWarning
from fedml.core.differential_privacy.mechanisms import PermuteAndFlip
from fedml.core.differential_privacy.validation import DiffprivlibMixin

Dataset = namedtuple('Dataset', ['X', 'y'])


class RandomForestClassifier(ForestClassifier, DiffprivlibMixin):
    r"""Random Forest Classifier with differential privacy.

    This class implements Differentially Private Random Decision Forests using Smooth Sensitivity [1].
    :math:`\epsilon`-Differential privacy is achieved by constructing decision trees via random splitting criterion and
    applying Exponential Mechanism to produce a noisy label.

    Parameters
    ----------
    n_estimators: int, default: 10
        The number of trees in the forest.

    epsilon: float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    cat_feature_threshold: int, default: 10
        Threshold value used to determine categorical features. For example, value of ``10`` means
        any feature that has less than or equal to 10 unique values will be treated as a categorical feature.

    n_jobs : int, default: 1
        Number of CPU cores used when parallelising over classes. ``-1`` means
        using all processors.

    verbose : int, default: 0
        Set to any positive number for verbosity.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    max_depth: int, default: 15
        The maximum depth of the tree. Final depth of the tree will be calculated based on the number of continuous
        and categorical features, but it wont be more than this number.
        Note: The depth translates to an exponential increase in memory usage.

    random_state: float, optional
        Sets the numpy random seed.

    feature_domains: dict, optional
        A dictionary of domain values for all features where keys are the feature indexes in the training data and
        the values are an array of domain values for categorical features and an array of min and max values for
        continuous features. For example, if the training data is [[2, 'dog'], [5, 'cat'], [7, 'dog']], then
        the feature_domains would be {'0': [2, 7], '1': ['dog', 'cat']}. If not provided, feature domains will
        be constructed from the data, but this will result in :class:`.PrivacyLeakWarning`.

    Attributes
    ----------
    n_features_in_: int
        The number of features when fit is performed.

    n_classes_: int
        The number of classes.

    classes_: array of shape (n_classes, )
        The classes labels.

    cat_features_: array of categorical feature indexes
        Categorical feature indexes.

    max_depth_: int
        Final max depth used for constructing decision trees.

    estimators_: list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    feature_domains_: dictionary of domain values mapped to feature
        indexes in the training data

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from fedml.core.differential_privacy.models import RandomForestClassifier
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]

    References
    ----------
    [1] Sam Fletcher, Md Zahidul Islam. "Differentially Private Random Decision Forests using Smooth Sensitivity"
    https://arxiv.org/abs/1606.03572

    """

    def __init__(self, n_estimators=10, *, epsilon=1.0, cat_feature_threshold=10, n_jobs=1, verbose=0, accountant=None,
                 max_depth=15, random_state=None, feature_domains=None, **unused_args):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("cat_feature_threshold", "max_depth", "epsilon", "random_state"),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.epsilon = epsilon
        self.cat_feature_threshold = cat_feature_threshold
        self.max_depth = max_depth
        self.accountant = BudgetAccountant.load_default(accountant)
        self.feature_domains = feature_domains

        if random_state is not None:
            np.random.seed(random_state)

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None):
        """Fit the model to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self: class

        """
        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        if not isinstance(self.n_estimators, numbers.Integral) or self.n_estimators < 0:
            raise ValueError(f'Number of estimators should be a positive integer; got {self.n_estimators}')

        if not isinstance(self.cat_feature_threshold, numbers.Integral) or self.cat_feature_threshold < 0:
            raise ValueError('Categorical feature threshold should be a positive integer;'
                             f'got {self.cat_feature_threshold}')

        self.accountant.check(self.epsilon, 0)

        X, y = self._validate_data(X, y, multi_output=False)

        self.n_outputs_ = 1
        self.n_features_in_ = X.shape[1]
        self.cat_features_ = get_cat_features(X, self.cat_feature_threshold)
        self.max_depth_ = calc_tree_depth(n_cont_features=self.n_features_in_-len(self.cat_features_),
                                          n_cat_features=len(self.cat_features_), max_depth=self.max_depth)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.feature_domains_ = self.feature_domains

        if self.feature_domains_ is None:
            warnings.warn(
                "`feature_domains` parameter hasn't been specified, "
                "so falling back to determining domains from the data.\n"
                "This may result in additional privacy leakage. To ensure differential privacy with no "
                "additional privacy loss, specify `feature_domains` according to the documentation",
                PrivacyLeakWarning)
            self.feature_domains_ = get_feature_domains(X, self.cat_features_)

        if len(self.feature_domains_) != self.n_features_in_:
            raise ValueError("Missing domains for some features in `feature_domains`")

        if self.n_estimators > len(X):
            raise ValueError('Number of estimators is more than the available samples')

        subset_size = int(len(X) / self.n_estimators)
        datasets = []
        estimators = []

        for i in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=self.max_depth_,
                                               epsilon=self.epsilon,
                                               feature_domains=self.feature_domains_,
                                               cat_features=self.cat_features_,
                                               classes=self.classes_)
            estimators.append(estimator)
            datasets.append(Dataset(X=X[i*subset_size:(i+1)*subset_size], y=y[i*subset_size:(i+1)*subset_size]))

        estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='processes')(
            delayed(lambda estimator, X, y: estimator.fit(X, y))(estimator, dataset.X, dataset.y)
            for estimator, dataset in zip(estimators, datasets)
        )

        self.estimators_ = estimators
        self.accountant.spend(self.epsilon, 0)
        self.fitted_ = True

        return self


class DecisionTreeClassifier(BaseDecisionTreeClassifier, DiffprivlibMixin):
    r"""Decision Tree Classifier with differential privacy.

    This class implements the base differentially private decision tree classifier
    for the Random Forest classifier algorithm. Not meant to be used separately.

    Parameters
    ----------
    epsilon: float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    cat_feature_threshold: int, default: 10
        Threshold value used to determine categorical features. For example, value of ``10`` means
        any feature that has less than or equal to 10 unique values will be treated as a categorical feature.

    max_depth: int, default: 15
        The maximum depth of the tree.

    random_state: float, optional
        Sets the numpy random seed.

    cat_features: array, optional
        Array of categorical feature indexes. If not provided, will be determined from the data based on the
        cat_feature_threshold.

    classes: array of shape (n_classes_, ), optional
        Array of class labels. If not provided, will be determined from the data.

    feature_domains: dict, optional
        A dictionary of domain values for all features where keys are the feature indexes in the training data and
        the values are an array of domain values for categorical features and an array of min and max values for
        continuous features. For example, if the training data is [[2, 'dog'], [5, 'cat'], [7, 'dog']], then
        the feature_domains would be {'0': [2, 7], '1': ['dog', 'cat']}. If not provided, feature domains will
        be constructed from the data, but this will result in :class:`.PrivacyLeakWarning`.

    Attributes
    ----------
    n_features_in_: int
        The number of features when fit is performed.

    n_classes_: int
        The number of classes.

    classes_: array of shape (n_classes, )
        The class labels.

    cat_features_: array of categorical feature indexes
        Categorical feature indexes.

    feature_domains_: dictionary of domain values mapped to feature
        indexes in the training data

    """
    def __init__(self, cat_feature_threshold=10, max_depth=15, epsilon=1, random_state=None, feature_domains=None,
                 cat_features=None, classes=None):
        # TODO: Remove try...except when sklearn v1.0 is min-requirement
        try:
            super().__init__(
                criterion=None,
                splitter=None,
                max_depth=max_depth,
                min_samples_split=None,
                min_samples_leaf=None,
                min_weight_fraction_leaf=None,
                max_features=None,
                random_state=random_state,
                max_leaf_nodes=None,
                min_impurity_decrease=None,
                min_impurity_split=None
            )
        except TypeError:
            super().__init__(
                criterion=None,
                splitter=None,
                max_depth=max_depth,
                min_samples_split=None,
                min_samples_leaf=None,
                min_weight_fraction_leaf=None,
                max_features=None,
                random_state=random_state,
                max_leaf_nodes=None,
                min_impurity_decrease=None
            )
        self.feature_domains = feature_domains
        self.cat_feature_threshold = cat_feature_threshold
        self.epsilon = epsilon
        self.cat_features = cat_features
        self.classes = classes

        if random_state is not None:
            np.random.seed(random_state)

    def _build(self, features, feature_domains, current_depth=1):
        if not features or current_depth >= self.max_depth+1:
            return DecisionNode(level=current_depth, classes=self.classes_)

        split_feature = np.random.choice(features)
        node = DecisionNode(level=current_depth, classes=self.classes_, split_feature=split_feature)

        if split_feature in self.cat_features_:
            node.set_split_type(DecisionNode.CAT_SPLIT)
            for value in feature_domains[str(split_feature)]:
                child_node = self._build([f for f in features if f != split_feature], feature_domains, current_depth+1)
                node.add_cat_child(value, child_node)
        else:
            node.set_split_type(DecisionNode.CONT_SPLIT)
            split_value = np.random.uniform(feature_domains[str(split_feature)][0],
                                            feature_domains[str(split_feature)][1])
            node.set_split_value(split_value)
            left_domain = {k: v if k != str(split_feature) else [v[0], split_value]
                           for k, v in feature_domains.items()}
            right_domain = {k: v if k != str(split_feature) else [split_value, v[1]]
                            for k, v in feature_domains.items()}
            left_child = self._build(features, left_domain, current_depth+1)
            right_child = self._build(features, right_domain, current_depth+1)
            node.set_left_child(left_child)
            node.set_right_child(right_child)

        return node

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"):
        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        if not isinstance(self.cat_feature_threshold, numbers.Integral) or self.cat_feature_threshold < 0:
            raise ValueError('Categorical feature threshold should be a positive integer;'
                             f'got {self.cat_feature_threshold}')

        if check_input:
            X, y = self._validate_data(X, y, multi_output=False)
        self.n_outputs_ = 1

        self.feature_domains_ = self.feature_domains
        self.cat_features_ = self.cat_features

        if self.cat_features_ is None:
            self.cat_features_ = get_cat_features(X, self.cat_feature_threshold)

        if self.feature_domains_ is None:
            warnings.warn(
                "feature_domains parameter hasn't been specified, "
                "so falling back to determining domains from the data.\n"
                "This may result in additional privacy leakage. To ensure differential privacy with no "
                "additional privacy loss, specify `feature_domains` according to the documentation",
                PrivacyLeakWarning)
            self.feature_domains_ = get_feature_domains(X, self.cat_features_)

        self.classes_ = self.classes

        if self.classes_ is None:
            self.classes_ = np.unique(y)

        self.n_classes_ = len(self.classes_)

        self.n_features_in_ = X.shape[1]
        features = list(range(self.n_features_in_))

        self.tree_ = self._build(features, self.feature_domains_)

        for i, _ in enumerate(X):
            node = self.tree_.classify(X[i])
            node.update_class_count(y[i].item())

        self.tree_.set_noisy_label(self.epsilon, self.classes_)

        return self

    @property
    def n_features_(self):
        return self.n_features_in_

    def _more_tags(self):
        return {}


class DecisionNode:
    """Base Decision Node
    """
    CONT_SPLIT = 0
    CAT_SPLIT = 1

    def __init__(self, level, classes, split_feature=None, split_value=None, split_type=None):
        """
        Initialize DecisionNode

        Parameters
        ----------
        level: int
            Node level in the tree

        classes: list
            List of class labels

        split_feature: int
            Split feature index

        split_value: Any
            Feature value to split at

        split_type: int
            Type of split

        """
        self._level = level
        self._classes = classes
        self._split_type = split_type
        self._split_feature = split_feature
        self._split_value = split_value
        self._left_child = None
        self._right_child = None
        self._cat_children = {}
        self._class_counts = defaultdict(int)
        self._noisy_label = None

    @property
    def noisy_label(self):
        """Get noisy label"""
        return self._noisy_label

    def set_split_value(self, split_value):
        """Set split value"""
        self._split_value = split_value

    def set_split_type(self, split_type):
        """Set split type"""
        self._split_type = split_type

    def set_left_child(self, node):
        """Set left child of the node"""
        self._left_child = node

    def set_right_child(self, node):
        """Set right child of the node"""
        self._right_child = node

    def add_cat_child(self, cat_value, node):
        """Add a categorical child node"""
        self._cat_children[str(cat_value)] = node

    def is_leaf(self):
        """Check whether the node is leaf node"""
        return not self._left_child and not self._right_child and not self._cat_children

    def update_class_count(self, class_value):
        """Update the class count for the given class"""
        self._class_counts[class_value] += 1

    def classify(self, x):
        """Classify the given data"""
        if self.is_leaf():
            return self

        child = None

        if self._split_type == self.CAT_SPLIT:
            x_val = str(x[self._split_feature])
            child = self._cat_children.get(x_val)
        else:
            x_val = x[self._split_feature]
            if x_val < self._split_value:
                child = self._left_child
            else:
                child = self._right_child

        if child is None:
            return self

        return child.classify(x)

    def set_noisy_label(self, epsilon, class_values):
        """Set the noisy label for this node"""
        if self.is_leaf():
            if not self._noisy_label:
                for val in class_values:
                    if val not in self._class_counts:
                        self._class_counts[val] = 0

                utility = list(self._class_counts.values())
                candidates = list(self._class_counts.keys())
                mech = PermuteAndFlip(epsilon=epsilon, sensitivity=1, monotonic=True, utility=utility,
                                      candidates=candidates)
                self._noisy_label = mech.randomise()
        else:
            if self._left_child:
                self._left_child.set_noisy_label(epsilon, class_values)
            if self._right_child:
                self._right_child.set_noisy_label(epsilon, class_values)
            for child_node in self._cat_children.values():
                child_node.set_noisy_label(epsilon, class_values)

    def predict(self, X):
        """Predict using this node"""
        y = []
        X = np.array(X)
        check_array(X)

        for x in X:
            node = self.classify(x)
            proba = np.zeros(len(self._classes))
            proba[np.where(self._classes == node.noisy_label)[0].item()] = 1
            y.append(proba)

        return np.array(y)


def get_feature_domains(X, cat_features):
    """Calculate feature domains from the data.

    Parameters:
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        cat_features : array of integers
            List of categorical feature indexes

    Returns:
        [dict]: Dictionary with keys as feature indexes and values as feature domains.
    """
    feature_domains = {}
    X_t = np.transpose(X)
    cont_features = list(set(range(X.shape[1])) - set(cat_features))

    for i in cat_features:
        feature_domains[str(i)] = [str(x) for x in set(X_t[i])]

    for i in cont_features:
        vals = [float(x) for x in X_t[i]]
        feature_domains[str(i)] = [min(vals), max(vals)]

    return feature_domains


def get_cat_features(X, feature_threshold=2):
    """Determine categorical features

    Parameters:
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        feature_threshold: int, defaults to 2.
            Threshold value used to determine categorical features. For example, value of ``10`` means
            any feature that has less than or equal to 10 unique values will be treated as a categorical feature.

    Returns:
        [list]: List of categorical feature indexes
    """
    n_features = X.shape[1]
    cat_features = []

    for i in range(n_features):
        values = set(X[:, i])
        if len(values) <= feature_threshold:
            cat_features.append(i)

    return cat_features


def calc_tree_depth(n_cont_features, n_cat_features, max_depth=15):
    """Calculate tree depth

    Args:
        n_cont_features (int): Number of continuous features
        n_cat_features ([type]): Number of categorical features
        max_depth (int, optional): Max depth tree. Defaults to 15.

    Returns:
        [int]: Final depth tree
    """
    if n_cont_features < 1:
        return min(max_depth, np.floor(n_cat_features / 2.))
    # Designed using balls-in-bins probability. See the paper for details.
    m = float(n_cont_features)
    depth = 0
    expected_empty = m   # the number of unique attributes not selected so far
    while expected_empty > m / 2.:   # repeat until we have less than half the attributes being empty
        expected_empty = m * ((m - 1.) / m) ** depth
        depth += 1
    # the above was only for half the numerical attributes. now add half the categorical attributes
    final_depth = np.floor(depth + (n_cat_features / 2.))
    return min(max_depth, final_depth)
