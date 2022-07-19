from unittest import TestCase

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from fedml.core.differential_privacy.models.naive_bayes import GaussianNB
from fedml.core.differential_privacy.utils import global_seed, PrivacyLeakWarning, DiffprivlibCompatibilityWarning, BudgetError


class TestGaussianNB(TestCase):
    def test_not_none(self):
        clf = GaussianNB(epsilon=1, bounds=(0, 1))
        self.assertIsNotNone(clf)

    def test_zero_epsilon(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB(epsilon=0, bounds=(0, 1))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_neg_epsilon(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB(epsilon=-1, bounds=(0, 1))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_sample_weight_warning(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]))
        w = abs(np.random.randn(10))

        with self.assertWarns(DiffprivlibCompatibilityWarning):
            clf.fit(X, y, sample_weight=w)

    def test_mis_ordered_bounds(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=([0, 1], [1, 0]))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_no_bounds(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB()

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X, y)

        self.assertIsNotNone(clf)

    def test_missing_bounds(self):
        X = np.random.random((10, 3))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_noisy_count(self):
        y = np.random.randint(20, size=10000)
        actual_counts = np.array([(y == y_i).sum() for y_i in np.unique(y)])

        clf = GaussianNB(epsilon=3)
        noisy_counts = clf._noisy_class_counts(y)
        self.assertEqual(y.shape[0], noisy_counts.sum())
        self.assertFalse(np.all(noisy_counts == actual_counts))

        clf = GaussianNB(epsilon=float("inf"))
        noisy_counts = clf._noisy_class_counts(y)
        self.assertEqual(y.shape[0], noisy_counts.sum())
        self.assertTrue(np.all(noisy_counts == actual_counts))

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_different_results(self):
        from sklearn.naive_bayes import GaussianNB as sk_nb
        from sklearn import datasets

        global_seed(12345)
        dataset = datasets.load_iris()

        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=.2)

        bounds = ([4.3, 2.0, 1.0, 0.1], [7.9, 4.4, 6.9, 2.5])

        clf_dp = GaussianNB(epsilon=1.0, bounds=bounds)
        clf_non_private = sk_nb()

        for clf in [clf_dp, clf_non_private]:
            clf.fit(x_train, y_train)

        # Todo: remove try...except when sklearn v1.0 is required
        try:
            nonprivate_var = clf_non_private.var_
        except AttributeError:
            nonprivate_var = clf_non_private.sigma_

        theta_diff = (clf_dp.theta_ - clf_non_private.theta_) ** 2
        self.assertGreater(theta_diff.sum(), 0)

        var_diff = (clf_dp.var_ - nonprivate_var) ** 2
        self.assertGreater(var_diff.sum(), 0)

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_with_iris(self):
        global_seed(12345)
        from sklearn import datasets
        dataset = datasets.load_iris()

        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=.2)

        bounds = ([4.3, 2.0, 1.0, 0.1], [7.9, 4.4, 6.9, 2.5])

        clf = GaussianNB(epsilon=5.0, bounds=bounds)
        clf.fit(x_train, y_train)

        accuracy = clf.score(x_test, y_test)
        counts = clf.class_count_.copy()
        self.assertGreater(accuracy, 0.45)

        clf.partial_fit(x_train, y_train)
        new_counts = clf.class_count_
        self.assertEqual(np.sum(new_counts), np.sum(counts) * 2)

    def test_accountant(self):
        from fedml.core.differential_privacy.accountant import BudgetAccountant
        acc = BudgetAccountant()

        x_train = np.random.random((10, 2))
        y_train = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1.0, bounds=(0, 1), accountant=acc)
        clf.fit(x_train, y_train)
        self.assertEqual((1, 0), acc.total())

        with BudgetAccountant(1.5, 0) as acc2:
            clf = GaussianNB(epsilon=1.0, bounds=(0, 1))
            clf.fit(x_train, y_train)
            self.assertEqual((1, 0), acc2.total())

            with self.assertRaises(BudgetError):
                clf.fit(x_train, y_train)

    def test_priors(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]), priors=(0.75, 0.25))
        self.assertIsNotNone(clf.fit(X, y))

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]), priors=(1,))
        with self.assertRaises(ValueError):
            clf.fit(X, y)

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]), priors=(0.5, 0.7))
        with self.assertRaises(ValueError):
            clf.fit(X, y)

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]), priors=(-0.5, 1.5))
        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_bad_refit_shape(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]))
        clf.fit(X, y)

        X2 = np.random.random((10, 3))
        clf.bounds = ([0, 0, 0], [1, 1, 1])

        with self.assertRaises(ValueError):
            clf.partial_fit(X2, y)

    def test_bad_refit_classes(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]))
        clf.fit(X, y)

        X2 = np.random.random((10, 2))
        y2 = np.random.randint(3, size=10)

        with self.assertRaises(ValueError):
            clf.partial_fit(X2, y2)

    def test_update_mean_variance(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]))
        self.assertIsNotNone(clf._update_mean_variance(0, 0, 0, X, n_noisy=5))
        self.assertIsNotNone(clf._update_mean_variance(0, 0, 0, X, n_noisy=0))
        self.assertWarns(PrivacyLeakWarning, clf._update_mean_variance, 0, 0, 0, X)
        self.assertWarns(DiffprivlibCompatibilityWarning, clf._update_mean_variance, 0, 0, 0, X, n_noisy=1,
                         sample_weight=1)

    def test_sigma(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]))
        clf.fit(X, y)
        self.assertIsInstance(clf.sigma_, np.ndarray)
