from unittest import TestCase

import numpy as np
import pytest

try:
    import sklearn.decomposition._pca as sk_pca
except ImportError:
    import sklearn.decomposition.pca as sk_pca

from fedml.core.differential_privacy.models.pca import PCA
from fedml.core.differential_privacy.utils import PrivacyLeakWarning, DiffprivlibCompatibilityWarning, BudgetError


class TestPCA(TestCase):
    def test_not_none(self):
        clf = PCA(epsilon=1, centered=True, data_norm=1)
        self.assertIsNotNone(clf)

    def test_no_params(self):
        clf = PCA()

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        X = X[:, np.newaxis]

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X)

    def test_no_range(self):
        clf = PCA(data_norm=6)

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        X = X[:, np.newaxis]

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X)

    def test_centered(self):
        clf = PCA(data_norm=3, centered=True)

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        X = X[:, np.newaxis]
        X -= X.mean(axis=0)

        self.assertIsNotNone(clf.fit(X))

    def test_fit_transform(self):
        X = np.random.randn(1000, 5)
        X -= np.mean(X, axis=0)
        X /= np.linalg.norm(X, axis=1).max()

        clf = PCA(2, epsilon=5, centered=True, data_norm=1)
        self.assertIsNotNone(clf.fit_transform(X))

    def test_large_norm(self):
        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        X = X[:, np.newaxis]

        clf = PCA(data_norm=1.0, centered=True)

        self.assertIsNotNone(clf.fit(X))

    def test_solver_warning(self):
        with self.assertWarns(DiffprivlibCompatibilityWarning):
            PCA(svd_solver='full')

    def test_inf_epsilon(self):
        X = np.random.randn(250, 21)
        X -= np.mean(X, axis=0)
        X /= np.linalg.norm(X, axis=1).max()

        for components in range(1, 10):
            clf = PCA(components, epsilon=float("inf"), centered=True, data_norm=1)
            clf.fit(X)

            sk_clf = sk_pca.PCA(components, svd_solver='full')
            sk_clf.fit(X)

            self.assertAlmostEqual(clf.score(X), sk_clf.score(X))
            self.assertTrue(np.allclose(clf.get_precision(), sk_clf.get_precision()))
            self.assertTrue(np.allclose(clf.get_covariance(), sk_clf.get_covariance()))
            self.assertTrue(np.allclose(np.abs((clf.components_ / sk_clf.components_).sum(axis=1)), clf.n_features_))

    def test_big_epsilon(self):
        X = np.random.randn(5000, 10)
        X -= np.mean(X, axis=0)
        X /= np.linalg.norm(X, axis=1).max()

        for components in range(1, 3):
            clf = PCA(components * 3, epsilon=10, centered=True, data_norm=1)
            clf.fit(X)

            sk_clf = sk_pca.PCA(components * 3, svd_solver='full')
            sk_clf.fit(X)

            self.assertAlmostEqual(clf.score(X) / sk_clf.score(X), 1, places=2)

    def test_mle_components(self):
        X = np.random.randn(1000, 5)
        X -= np.mean(X, axis=0)
        X /= np.linalg.norm(X, axis=1).max()

        clf = PCA("mle", epsilon=5, centered=True, data_norm=1)
        self.assertIsNotNone(clf.fit(X))

    def test_float_components(self):
        X = np.random.randn(1000, 5)
        X -= np.mean(X, axis=0)
        X /= np.linalg.norm(X, axis=1).max()

        clf = PCA(0.5, epsilon=5, centered=True, data_norm=1)
        self.assertIsNotNone(clf.fit(X))

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_different_results(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        dataset = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

        mean = np.mean(X_train, axis=0)
        X_train -= mean
        X_test -= mean

        clf = PCA(data_norm=12, centered=True)
        clf.fit(X_train, y_train)

        transform1 = clf.transform(X_test)

        clf = PCA(data_norm=12, centered=True)
        clf.fit(X_train, y_train)

        transform2 = clf.transform(X_test)

        clf = sk_pca.PCA(svd_solver='full')
        clf.fit(X_train, y_train)

        transform3 = clf.transform(X_test)

        self.assertFalse(np.all(transform1 == transform2))
        self.assertFalse(np.all(transform3 == transform1) and np.all(transform3 == transform2))

    def test_accountant(self):
        from fedml.core.differential_privacy.accountant import BudgetAccountant
        acc = BudgetAccountant()

        X = np.random.randn(5000, 21)
        X -= np.mean(X, axis=0)
        X /= np.linalg.norm(X, axis=1).max()

        clf = PCA(5, epsilon=5, centered=True, data_norm=1, accountant=acc)
        clf.fit(X)

        self.assertEqual((5, 0), acc.total())

        with BudgetAccountant(8, 0) as acc2:
            clf = PCA(5, epsilon=5, centered=True, data_norm=1)
            clf.fit(X)
            self.assertEqual((5, 0), acc2.total())

            with self.assertRaises(BudgetError):
                clf.fit(X)
