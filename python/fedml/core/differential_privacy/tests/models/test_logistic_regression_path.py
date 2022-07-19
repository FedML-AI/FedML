from unittest import TestCase

import numpy as np

from fedml.core.differential_privacy.models.logistic_regression import _logistic_regression_path


class TestLogisticRegressionPath(TestCase):
    def setup_class(self):
        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        self.X = X
        self.y = y

    def test_not_none(self):
        self.assertIsNotNone(_logistic_regression_path)

    def test_no_epsilon(self):
        with self.assertRaises(TypeError):
            _logistic_regression_path(self.X, self.y, Cs=[1e5])

    def test_no_norm(self):
        with self.assertRaises(TypeError):
            _logistic_regression_path(self.X, self.y, Cs=[1e5], epsilon=1)

    def test_with_dataset(self):
        output = _logistic_regression_path(self.X, self.y, epsilon=1, data_norm=1, Cs=[1e5])

        self.assertIsInstance(output, tuple)

    def test_pos_class(self):
        X = np.array(
            [0.50, 0.75, 1.00])
        y = np.array([0, 1, 2])
        X = X[:, np.newaxis]

        with self.assertRaises(ValueError):
            _logistic_regression_path(X, y, epsilon=1, data_norm=1.0, Cs=[1e5], pos_class=None)

    def test_coef(self):
        _logistic_regression_path(self.X, self.y, epsilon=1, data_norm=1.0, Cs=[1e5], coef=np.ones(2))

        with self.assertRaises(ValueError):
            _logistic_regression_path(self.X, self.y, epsilon=1, data_norm=1.0, Cs=[1e5], coef=np.ones(3))

    def test_Cs(self):
        coefs, Cs, n_iter = _logistic_regression_path(self.X, self.y, epsilon=1, data_norm=1, Cs=3)

        self.assertEqual(len(coefs), 3)
        self.assertEqual(len(Cs), 3)
        self.assertEqual(len(n_iter), 3)

