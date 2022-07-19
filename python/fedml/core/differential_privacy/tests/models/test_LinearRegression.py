import numpy as np
from unittest import TestCase

import pytest

from fedml.core.differential_privacy.models.linear_regression import LinearRegression
from fedml.core.differential_privacy.models.logistic_regression import _check_solver, _check_multi_class
from fedml.core.differential_privacy.utils import global_seed, PrivacyLeakWarning, DiffprivlibCompatibilityWarning, BudgetError


class TestLinearRegression(TestCase):
    def setup_method(self, method):
        global_seed(3141592653)

    def test_not_none(self):
        self.assertIsNotNone(LinearRegression)

    def test_no_params(self):
        clf = LinearRegression()

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X, y)

    def test_sample_weight_warning(self):
        clf = LinearRegression(bounds_X=(0, 5), bounds_y=(0, 1))

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        with self.assertWarns(DiffprivlibCompatibilityWarning):
            clf.fit(X, y, sample_weight=np.ones_like(y))

    def test_no_bounds_y(self):
        clf = LinearRegression(bounds_X=(0, 5))

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X, y)

    def test_large_data(self):
        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        clf = LinearRegression(bounds_X=(0, 3), bounds_y=(0, 1), fit_intercept=False)

        self.assertIsNotNone(clf.fit(X, y))

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_different_results(self):
        from sklearn import datasets
        from sklearn import linear_model
        from sklearn.model_selection import train_test_split

        dataset = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

        clf = LinearRegression(bounds_X=([4.3, 2.0, 1.1, 0.1], [7.9, 4.4, 6.9, 2.5]), bounds_y=(0, 2))
        clf.fit(X_train, y_train)

        predict1 = clf.predict(X_test)

        clf = LinearRegression(bounds_X=([4.3, 2.0, 1.1, 0.1], [7.9, 4.4, 6.9, 2.5]), bounds_y=(0, 2))
        clf.fit(X_train, y_train)

        predict2 = clf.predict(X_test)

        clf = linear_model.LinearRegression()
        clf.fit(X_train, y_train)

        predict3 = clf.predict(X_test)

        self.assertFalse(np.all(predict1 == predict2))
        self.assertFalse(np.all(predict3 == predict1) and np.all(predict3 == predict2))

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_same_results(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn import linear_model

        dataset = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

        clf = LinearRegression(epsilon=float("inf"), bounds_X=([4.3, 2.0, 1.0, 0.1], [7.9, 4.4, 6.9, 2.5]),
                               bounds_y=(0, 2))
        clf.fit(X_train, y_train)

        predict1 = clf.predict(X_test)

        clf2 = linear_model.LinearRegression()
        clf2.fit(X_train, y_train)

        predict2 = clf2.predict(X_test)

        self.assertTrue(np.allclose(predict1, predict2))
        self.assertTrue(np.allclose(clf.coef_, clf2.coef_))

    def test_simple(self):
        X = np.linspace(-1, 1, 1000)
        y = X.copy()
        X = X[:, np.newaxis]

        clf = LinearRegression(epsilon=2, fit_intercept=False, bounds_X=(-1, 1), bounds_y=(-1, 1))
        clf.fit(X, y)
        print(clf.predict(np.array([0.5]).reshape(-1, 1)))

        self.assertIsNotNone(clf)
        self.assertAlmostEqual(clf.predict(np.array([0.5]).reshape(-1, 1))[0], 0.5, delta=.05)

    def test_copy(self):
        X = np.linspace(-1, 1, 1000)
        y = X.copy()
        X = X[:, np.newaxis]

        clf = LinearRegression(epsilon=2, fit_intercept=False, bounds_X=(-1, 1), bounds_y=(-1, 1), copy_X=True)
        clf.fit(X, y)

        self.assertIsNotNone(clf)
        self.assertIsNotNone(clf._preprocess_data(X, y, fit_intercept=True, bounds_X=(-1, 1), bounds_y=(-1, 1),
                                                  check_input=False, copy=True))

    def test_multiple_targets(self):
        from sklearn.linear_model import LinearRegression as sk_LinearRegression
        X = np.linspace(-1, 1, 1000)
        y = np.vstack((X.copy(), X.copy(), X.copy(), X.copy())).T
        X = X[:, np.newaxis]

        clf_dp = LinearRegression(epsilon=2, fit_intercept=False, bounds_X=(-1, 1), bounds_y=(-1, 1))

        with self.assertRaises(ValueError):
            clf_dp.fit(X, y.T)

        clf_dp.fit(X, y)

        clf_sk = sk_LinearRegression(fit_intercept=False)
        clf_sk.fit(X, y)

        x0 = np.array([[0.5]])

        self.assertEqual(clf_dp.coef_.shape, clf_sk.coef_.shape)
        self.assertEqual(clf_dp.predict(x0).shape, clf_sk.predict(x0).shape)

        clf_dp2 = LinearRegression(epsilon=2, fit_intercept=True, bounds_X=(-1, 1), bounds_y=(-1, 1)).fit(X, y)
        self.assertIsNotNone(clf_dp2)
        self.assertEqual(clf_dp2.intercept_.shape, (4,))
        self.assertEqual(clf_dp2.coef_.shape, clf_sk.coef_.shape)
        self.assertEqual(clf_dp2.predict(x0).shape, clf_sk.predict(x0).shape)

    def test_check_solver(self):
        with self.assertWarns(DiffprivlibCompatibilityWarning):
            _check_solver("wrong_solver", "l2", False)

        with self.assertRaises(ValueError):
            _check_solver("lbfgs", "l1", False)

        with self.assertRaises(ValueError):
            _check_solver("lbfgs", "l2", True)

        self.assertIsNotNone(_check_solver("lbfgs", "l2", False))

    def test_check_multi_class(self):
        with self.assertWarns(DiffprivlibCompatibilityWarning):
            _check_multi_class("multinomial", "lbfgs", 2)

        self.assertIsNotNone(_check_multi_class("ovr", "lbfgs", 2))

    def test_accountant(self):
        from fedml.core.differential_privacy.accountant import BudgetAccountant

        acc = BudgetAccountant()
        X = np.linspace(-1, 1, 100)
        y = X.copy()
        X = X[:, np.newaxis]

        clf = LinearRegression(epsilon=2, fit_intercept=False, bounds_X=(-1, 1), bounds_y=(-1, 1), accountant=acc)
        clf.fit(X, y)
        self.assertEqual((2, 0), acc.total())

        with BudgetAccountant(3, 0) as acc2:
            clf = LinearRegression(epsilon=2, fit_intercept=False, bounds_X=(-1, 1), bounds_y=(-1, 1))
            clf.fit(X, y)
            self.assertEqual((2, 0), acc2.total())

            with self.assertRaises(BudgetError):
                clf.fit(X, y)
