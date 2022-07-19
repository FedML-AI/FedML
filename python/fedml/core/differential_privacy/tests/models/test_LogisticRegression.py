import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.models.logistic_regression import LogisticRegression
from fedml.core.differential_privacy.utils import global_seed, PrivacyLeakWarning, DiffprivlibCompatibilityWarning, BudgetError


class TestLogisticRegression(TestCase):
    def setup_method(self, method):
        global_seed(3141592653)

    def test_not_none(self):
        self.assertIsNotNone(LogisticRegression)

    def test_bad_params(self):
        X = [[1]]
        y = [0]

        with self.assertRaises(ValueError):
            LogisticRegression(data_norm=1, C=-1).fit(X, y)

        with self.assertRaises(ValueError):
            LogisticRegression(data_norm=1, C=1.2).fit(X, y)

        with self.assertRaises(ValueError):
            LogisticRegression(data_norm=1, max_iter=-1).fit(X, y)

        with self.assertRaises(ValueError):
            LogisticRegression(data_norm=1, max_iter="100").fit(X, y)

        with self.assertRaises(ValueError):
            LogisticRegression(data_norm=1, tol=-1).fit(X, y)

        with self.assertRaises(ValueError):
            LogisticRegression(data_norm=1, tol="1").fit(X, y)

    def test_one_class(self):
        X = [[1]]
        y = [0]

        clf = LogisticRegression(data_norm=1)

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_no_params(self):
        clf = LogisticRegression()

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X, y)

    def test_smaller_norm_than_data(self):
        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        clf = LogisticRegression(data_norm=1.0)

        self.assertIsNotNone(clf.fit(X, y))

    def test_sample_weight_warning(self):
        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        clf = LogisticRegression(data_norm=5.5)

        with self.assertWarns(DiffprivlibCompatibilityWarning):
            clf.fit(X, y, sample_weight=np.ones_like(y))

    def test_warm_start(self):
        X = np.array(
            [0.50, 0.75, 1.00])
        y = np.array([0, 1, 2])
        X = X[:, np.newaxis]

        clf = LogisticRegression(data_norm=1.0, warm_start=True)
        clf.fit(X, y)
        self.assertIsNotNone(clf.fit(X, y))

    def test_trinomial(self):
        X = np.array(
            [0.50, 0.75, 1.00])
        y = np.array([0, 1, 2])
        X = X[:, np.newaxis]

        clf = LogisticRegression(data_norm=1.0)

        self.assertIsNotNone(clf.fit(X, y))

    def test_quadnomial(self):
        X = np.array(
            [0.25, 0.50, 0.75, 1.00])
        y = np.array([0, 1, 2, 3])
        X = X[:, np.newaxis]

        clf = LogisticRegression(data_norm=1.0)

        self.assertIsNotNone(clf.fit(X, y))

    def test_multi_dim_y(self):
        X = np.array(
            [0.25, 0.50, 0.75, 1.00])
        y = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        X = X[:, np.newaxis]

        clf = LogisticRegression(data_norm=1.0)

        self.assertRaises(ValueError, clf.fit, X, y)

    def test_solver_warning(self):
        with self.assertWarns(DiffprivlibCompatibilityWarning):
            LogisticRegression(solver="newton-cg")

    def test_multi_class_warning(self):
        with self.assertWarns(DiffprivlibCompatibilityWarning):
            LogisticRegression(multi_class="multinomial")

    def test_different_results(self):
        from sklearn import datasets
        from sklearn import linear_model
        from sklearn.model_selection import train_test_split

        dataset = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

        clf = LogisticRegression(data_norm=12)
        clf.fit(X_train, y_train)

        predict1 = clf.predict(X_test)

        clf = LogisticRegression(data_norm=12)
        clf.fit(X_train, y_train)

        predict2 = clf.predict(X_test)

        clf = linear_model.LogisticRegression(solver="lbfgs", multi_class="ovr")
        clf.fit(X_train, y_train)

        predict3 = clf.predict(X_test)

        self.assertTrue(np.any(predict1 != predict2) or np.any(predict1 != predict3))

    def test_same_results(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn import linear_model

        dataset = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

        clf = LogisticRegression(data_norm=12, epsilon=float("inf"))
        clf.fit(X_train, y_train)

        predict1 = clf.predict(X_test)

        clf = linear_model.LogisticRegression(solver="lbfgs", multi_class="ovr")
        clf.fit(X_train, y_train)

        predict2 = clf.predict(X_test)

        self.assertTrue(np.all(predict1 == predict2))

    def test_simple(self):
        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50] * 5)
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1] * 5)
        X = X[:, np.newaxis]
        X -= 3.0
        X /= 2.5

        clf = LogisticRegression(epsilon=2, data_norm=1.0)
        clf.fit(X, y)

        self.assertIsNotNone(clf)
        self.assertFalse(clf.predict(np.array([(0.5 - 3) / 2.5]).reshape(-1, 1)))
        self.assertTrue(clf.predict(np.array([(5.5 - 3) / 2.5]).reshape(-1, 1)))

    def test_accountant(self):
        from fedml.core.differential_privacy.accountant import BudgetAccountant
        acc = BudgetAccountant()

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]
        X -= 3.0
        X /= 2.5

        clf = LogisticRegression(epsilon=2, data_norm=1.0, accountant=acc)
        clf.fit(X, y)
        self.assertEqual((2, 0), acc.total())

        with BudgetAccountant(3, 0) as acc2:
            clf = LogisticRegression(epsilon=2, data_norm=1.0)
            clf.fit(X, y)
            self.assertEqual((2, 0), acc2.total())

            with self.assertRaises(BudgetError):
                clf.fit(X, y)
