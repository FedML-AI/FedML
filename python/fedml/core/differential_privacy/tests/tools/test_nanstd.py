from unittest import TestCase

import numpy as np

from fedml.core.differential_privacy.tools.utils import nanstd
from fedml.core.differential_privacy.utils import PrivacyLeakWarning, global_seed, BudgetError


class TestNanStd(TestCase):
    def test_not_none(self):
        mech = nanstd
        self.assertIsNotNone(mech)

    def test_no_params(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = nanstd(a)
        self.assertIsNotNone(res)

    def test_no_epsilon(self):
        a = np.array([1, 2, 3])
        self.assertIsNotNone(nanstd(a, bounds=(0, 1)))

    def test_no_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            nanstd(a, epsilon=1)

    def test_bad_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            nanstd(a, epsilon=1, bounds=(0, -1))

    def test_missing_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = nanstd(a, 1, None)
        self.assertIsNotNone(res)

    def test_large_epsilon(self):
        global_seed(12345)
        a = np.random.random(1000)
        res = float(np.std(a))
        res_dp = nanstd(a, epsilon=5, bounds=(0, 1))

        self.assertAlmostEqual(res, res_dp, delta=0.01)

    def test_large_epsilon_axis(self):
        global_seed(12345)
        a = np.random.random((1000, 5))
        res = np.std(a, axis=0)
        res_dp = nanstd(a, epsilon=15, bounds=(0, 1), axis=0)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i], res_dp[i], delta=0.01)

    def test_array_like(self):
        self.assertIsNotNone(nanstd([1, 2, 3], bounds=(1, 3)))
        self.assertIsNotNone(nanstd((1, 2, 3), bounds=(1, 3)))

    def test_clipped_output(self):
        a = np.random.random((10,))

        for i in range(100):
            self.assertTrue(0 <= nanstd(a, epsilon=1e-5, bounds=(0, 1)) <= 1)

    def test_nan(self):
        a = np.random.random((5, 5))
        a[2, 2] = np.nan

        res = nanstd(a, bounds=(0, 1))
        self.assertFalse(np.isnan(res))

    def test_accountant(self):
        from fedml.core.differential_privacy.accountant import BudgetAccountant
        acc = BudgetAccountant(1.5, 0)

        a = np.random.random((1000, 5))
        nanstd(a, epsilon=1, bounds=(0, 1), accountant=acc)
        self.assertEqual((1.0, 0), acc.total())

        with acc:
            with self.assertRaises(BudgetError):
                nanstd(a, epsilon=1, bounds=(0, 1))
