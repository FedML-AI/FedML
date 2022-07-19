import numpy as np
from unittest import TestCase
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from fedml.core.differential_privacy.models.forest import DecisionTreeClassifier, get_cat_features, get_feature_domains, calc_tree_depth
from fedml.core.differential_privacy.utils import PrivacyLeakWarning, global_seed, DiffprivlibCompatibilityWarning


class TestDecisionTreeClassifier(TestCase):
    def setUp(self):
        global_seed(2718281828)

    def test_not_none(self):
        self.assertIsNotNone(DecisionTreeClassifier)

    def test_bad_params(self):
        X = [[1]]
        y = [0]
        
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(cat_feature_threshold="5").fit(X, y)

        with self.assertWarns(DiffprivlibCompatibilityWarning):
            DecisionTreeClassifier(cat_feature_threshold=2, feature_domains={'0': [0, 1]}).fit(X, y, sample_weight=1)

    def test_bad_data(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(feature_domains={'0': [0, 1]}).fit([[1]], None)

        with self.assertRaises(ValueError):
            DecisionTreeClassifier(feature_domains={'0': [0, 2]}).fit([[1], [2]], [1])

        with self.assertRaises(ValueError):
            DecisionTreeClassifier(feature_domains={'0': [0, 2]}).fit([[1], [2]], [[1, 2], [2, 4]])

    def test_simple(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)
        model = DecisionTreeClassifier(epsilon=5, cat_feature_threshold=2, max_depth=5, random_state=25)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        # when `feature_domains` is not provided, we should get a privacy leakage warning
        with self.assertWarns(PrivacyLeakWarning):
            model.fit(X, y)
        check_is_fitted(model)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))

    def test_with_feature_domains(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)
        model = DecisionTreeClassifier(epsilon=5, cat_feature_threshold=2, max_depth=5, random_state=25,
                                       feature_domains={'0': [2.0, 12.0], '1': [3.0, 13.0], '2': [4.0, 15.0]})
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        model.fit(X, y)
        check_is_fitted(model)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))

    def test_with_non_binary_labels(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([3, 3, 3, 3, 5, 5, 3] * 3)
        model = DecisionTreeClassifier(epsilon=5, cat_feature_threshold=2, max_depth=5, random_state=25,
                                       feature_domains={'0': [2.0, 12.0], '1': [3.0, 13.0], '2': [4.0, 15.0]})
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        model.fit(X, y)
        check_is_fitted(model)
        self.assertEqual(model.predict(np.array([[12, 3, 14]])), 3)


class TestUtils(TestCase):
    def test_calc_tree_depth(self):
        self.assertEqual(calc_tree_depth(0, 4), 2)
        self.assertEqual(calc_tree_depth(0, 100, max_depth=20), 20)
        self.assertEqual(calc_tree_depth(4, 5), 6)
        self.assertEqual(calc_tree_depth(4, 5, max_depth=3), 3)
        self.assertEqual(calc_tree_depth(40, 50), 15)

    def test_get_feature_domains(self):
        X = np.array([[12, 3, 14, 21], [0.1, 0.5, 0.7, 1], ['cat', 'dog', 'mouse', 'cat'], [0, 1, 0, 1]]).T
        cat_features = [2, 3]
        feature_domains = get_feature_domains(X, cat_features)
        self.assertEqual(feature_domains['0'], [3, 21])
        self.assertEqual(feature_domains['1'], [0.1, 1])
        self.assertEqual(set(feature_domains['2']), set(['dog', 'cat', 'mouse']))
        self.assertEqual(set(feature_domains['3']), set(['0', '1']))
    
    def test_get_cat_features(self):
        X = np.array([[12, 3, 14, 21], [0.1, 0.5, 0.7, 1], ['cat', 'dog', 'mouse', 'cat'], [0, 1, 0, 1]]).T
        cat_features = get_cat_features(X)
        self.assertEqual(cat_features, [3])
        cat_features = get_cat_features(X, feature_threshold=3)
        self.assertEqual(cat_features, [2, 3])