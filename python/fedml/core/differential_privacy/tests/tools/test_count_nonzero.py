from unittest import TestCase

import numpy as np

from fedml.core.differential_privacy.tools.utils import count_nonzero
from fedml.core.differential_privacy.utils import BudgetError


class TestCountNonZero(TestCase):
    def test_not_none(self):
        mech = count_nonzero
        self.assertIsNotNone(mech)

    def test_no_params(self):
        a = np.array([1, 2, 3])
        res = count_nonzero(a)
        self.assertIsNotNone(res)

    def test_no_epsilon(self):
        a = np.array([1, 2, 3])
        self.assertIsNotNone(count_nonzero(a))

    def test_inf_epsilon(self):
        a = np.random.random(1000).round()
        res = float(np.count_nonzero(a))
        res_dp = count_nonzero(a, epsilon=float("inf"))

        self.assertAlmostEqual(res, res_dp)

    def test_large_epsilon(self):
        a = np.random.random(1000).round()
        res = float(np.count_nonzero(a))
        res_dp = count_nonzero(a, epsilon=5)

        self.assertAlmostEqual(res, res_dp, delta=0.01 * res)

    def test_array_like(self):
        self.assertIsNotNone(count_nonzero([-1, 0, 1]))
        self.assertIsNotNone(count_nonzero((-1, 0, 1)))

    def test_axis(self):
        a = np.random.random((1000, 5)).round()
        res_dp = count_nonzero(a, epsilon=1, axis=0)
        self.assertEqual(res_dp.shape, (5,))

    def test_strings(self):
        a = ["", "", "Diffprivlib", "Diffprivlib", "Python", "Numpy"]
        res = np.count_nonzero(a)
        res_dp = count_nonzero(a, epsilon=float("inf"))

        self.assertEqual(res, res_dp)

    def test_nan(self):
        a = np.random.random((5, 5)).round()
        a[2, 2] = np.nan

        res = count_nonzero(a)
        self.assertFalse(np.isnan(res))

    def test_keepdims(self):
        a = np.random.random((5, 5)).round()

        self.assertEqual(count_nonzero(a, axis=0).ndim, 1)
        self.assertEqual(count_nonzero(a, axis=0, keepdims=True).ndim, 2)
        self.assertEqual(count_nonzero(a, axis=0, keepdims=True).shape, (1, 5))

    def test_accountant(self):
        from fedml.core.differential_privacy.accountant import BudgetAccountant
        acc = BudgetAccountant(1.5, 0)

        a = np.random.random((1000, 5)).round()
        count_nonzero(a, epsilon=1, accountant=acc)
        self.assertEqual((1.0, 0), acc.total())

        with acc:
            with self.assertRaises(BudgetError):
                count_nonzero(a, epsilon=1)
