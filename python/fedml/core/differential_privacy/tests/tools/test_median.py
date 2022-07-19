from unittest import TestCase

import numpy as np

from fedml.core.differential_privacy.tools.quantiles import median
from fedml.core.differential_privacy.utils import PrivacyLeakWarning, BudgetError


class TestMedian(TestCase):
    def test_not_none(self):
        mech = median
        self.assertIsNotNone(mech)

    def test_no_params(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = median(a)
        self.assertIsNotNone(res)

    def test_no_epsilon(self):
        a = np.array([1, 2, 3])
        self.assertIsNotNone(median(a, bounds=(0, 1)))

    def test_no_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = median(a, epsilon=1)
        self.assertIsNotNone(res)

    def test_bad_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            median(a, epsilon=1, bounds=(0, -1))

    def test_missing_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = median(a, epsilon=1, bounds=None)
        self.assertIsNotNone(res)

    def test_large_epsilon(self):
        a = np.random.random(1000)
        res = np.median(a)
        res_dp = median(a, epsilon=5, bounds=(0, 1))

        self.assertAlmostEqual(float(res), float(res_dp), delta=0.01)

    def test_inf_epsilon(self):
        a = [0, 1, 2, 3]

        for _ in range(100):
            res_dp = median(a, epsilon=float("inf"), bounds=(0, 3))
            self.assertTrue(1 <= res_dp <= 2)

    def test_output_type(self):
        res = median([1, 2, 3], bounds=(1, 3))
        self.assertTrue(isinstance(res, float))

    def test_simple(self):
        a = np.random.random(1000)

        res = median(a, epsilon=5, bounds=(0, 1))
        self.assertAlmostEqual(res, 0.5, delta=0.05)

    def test_normal(self):
        a = np.random.normal(size=2000)
        res = median(a, epsilon=3, bounds=(-3, 3))
        self.assertAlmostEqual(res, 0, delta=0.1)

    def test_axis(self):
        a = np.random.random((10, 5))
        out = median(a, epsilon=1, bounds=(0, 1), axis=0)
        self.assertEqual(out.shape, (5,))

        out = median(a, epsilon=1, bounds=(0, 1), axis=())
        self.assertEqual(out.shape, a.shape)

    def test_keepdims(self):
        a = np.random.random((10, 5))
        out = median(a, epsilon=1, bounds=(0, 1), keepdims=True)
        self.assertEqual(a.ndim, out.ndim)

    def test_array_like(self):
        self.assertIsNotNone(median([1, 2, 3], bounds=(1, 3)))
        self.assertIsNotNone(median((1, 2, 3), bounds=(1, 3)))

    def test_clipped_output(self):
        a = np.random.random((10,))

        for i in range(100):
            self.assertTrue(0 <= median(a, epsilon=1e-5, bounds=(0, 1)) <= 1)

    def test_nan(self):
        a = np.random.random((5, 5))
        a[2, 2] = np.nan

        res = median(a, bounds=(0, 1))
        self.assertTrue(np.isnan(res))

    def test_accountant(self):
        from fedml.core.differential_privacy.accountant import BudgetAccountant
        acc = BudgetAccountant(1.5, 0)

        a = np.random.random((1000, 5))
        median(a, epsilon=1, bounds=(0, 1), accountant=acc)
        self.assertEqual((1.0, 0), acc.total())

        with acc:
            with self.assertRaises(BudgetError):
                median(a, epsilon=1, bounds=(0, 1))

    def test_accountant_with_axes(self):
        from fedml.core.differential_privacy.accountant import BudgetAccountant
        acc = BudgetAccountant()

        a = np.random.random((1000, 4))
        median(a, epsilon=1, bounds=(0, 1), axis=0, accountant=acc)

        # Expecting a different spend on each of the 8 outputs
        self.assertEqual((1, 0), acc.total())
        self.assertEqual(4, len(acc))
