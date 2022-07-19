import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.accountant import BudgetAccountant
from fedml.core.differential_privacy.tools.histograms import histogram
from fedml.core.differential_privacy.utils import global_seed, BudgetError


class TestHistogram(TestCase):
    def test_no_params(self):
        a = np.array([1, 2, 3, 4, 5])
        with self.assertWarns(RuntimeWarning):
            res = histogram(a)
        self.assertIsNotNone(res)

    def test_no_range(self):
        a = np.array([1, 2, 3, 4, 5])
        with self.assertWarns(RuntimeWarning):
            res = histogram(a, epsilon=1)
        self.assertIsNotNone(res)

    def test_same_edges(self):
        a = np.array([1, 2, 3, 4, 5])
        _, edges = np.histogram(a, bins=3, range=(0, 10))
        _, dp_edges = histogram(a, epsilon=1, bins=3, range=(0, 10))
        self.assertTrue((edges == dp_edges).all())

    def test_different_result(self):
        global_seed(3141592653)
        a = np.array([1, 2, 3, 4, 5])
        hist, _ = np.histogram(a, bins=3, range=(0, 10))
        dp_hist, _ = histogram(a, epsilon=0.01, bins=3, range=(0, 10))

        # print("Non-private histogram: %s" % hist)
        # print("Private histogram: %s" % dp_hist)
        self.assertTrue((hist != dp_hist).any())

    def test_density(self):
        global_seed(3141592653)
        a = np.array([1, 2, 3, 4, 5])
        dp_hist, _ = histogram(a, epsilon=10, bins=3, range=(0, 10), density=True)

        self.assertAlmostEqual(dp_hist.sum(), 3 / 10)

    def test_accountant(self):
        acc = BudgetAccountant(1.5, 0)

        a = np.array([1, 2, 3, 4, 5])
        histogram(a, epsilon=1, bins=3, range=(0, 10), density=True, accountant=acc)
        self.assertEqual((1, 0), acc.total())

        with self.assertRaises(BudgetError):
            histogram(a, epsilon=1, bins=3, range=(0, 10), density=True, accountant=acc)

        with self.assertRaises(TypeError):
            histogram(a, epsilon=1, bins=3, range=(0, 10), density=True, accountant=[acc])

    def test_default_accountant(self):
        BudgetAccountant.pop_default()

        a = np.array([1, 2, 3, 4, 5])
        histogram(a, epsilon=1, bins=3, range=(0, 10), density=True)
        acc = BudgetAccountant.pop_default()
        self.assertEqual((1, 0), acc.total())

        histogram(a, epsilon=1, bins=3, range=(0, 10))
        acc2 = BudgetAccountant.pop_default()
        self.assertEqual((1, 0), acc.total())
        self.assertIsNot(acc, acc2)
