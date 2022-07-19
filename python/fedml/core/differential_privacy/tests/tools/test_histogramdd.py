import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.accountant import BudgetAccountant
from fedml.core.differential_privacy.tools.histograms import histogramdd
from fedml.core.differential_privacy.utils import global_seed, PrivacyLeakWarning, BudgetError


class TestHistogramdd(TestCase):
    def test_no_params(self):
        a = np.array([1, 2, 3, 4, 5])
        with self.assertWarns(PrivacyLeakWarning):
            res = histogramdd(a)
        self.assertIsNotNone(res)

    def test_no_range(self):
        a = np.array([1, 2, 3, 4, 5])
        with self.assertWarns(PrivacyLeakWarning):
            res = histogramdd(a, epsilon=2)
        self.assertIsNotNone(res)

    def test_bins_instead_of_range(self):
        a = np.array([1, 2, 3, 4, 5])
        res = histogramdd([a, a], epsilon=2, bins=([0, 2, 6], [0, 2, 6]))
        self.assertIsNotNone(res)

    def test_same_edges(self):
        a = np.array([1, 2, 3, 4, 5])
        _, edges = np.histogramdd(a, bins=3, range=[(0, 10)])
        _, dp_edges = histogramdd(a, epsilon=1, bins=3, range=[(0, 10)])

        for i in range(len(edges)):
            self.assertTrue((edges[i] == dp_edges[i]).all())

    def test_different_result(self):
        global_seed(3141592653)
        a = np.array([1, 2, 3, 4, 5])
        hist, _ = np.histogramdd(a, bins=3, range=[(0, 10)])
        dp_hist, _ = histogramdd(a, epsilon=0.1, bins=3, range=[(0, 10)])

        # print("Non-private histogram: %s" % hist)
        # print("Private histogram: %s" % dp_hist)
        self.assertTrue((hist != dp_hist).any())

    def test_density_1d(self):
        global_seed(3141592653)
        a = np.array([1, 2, 3, 4, 5])
        dp_hist, _ = histogramdd(a, epsilon=10, bins=3, range=[(0, 10)], density=True)

        # print(dp_hist.sum())

        self.assertAlmostEqual(dp_hist.sum(), 1.0 * 3 / 10)

    def test_density_2d(self):
        global_seed(3141592653)
        a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T
        dp_hist, _ = histogramdd(a, epsilon=10, bins=3, range=[(0, 10), (0, 10)], density=True)

        # print(dp_hist.sum())

        self.assertAlmostEqual(dp_hist.sum(), 1.0 * (3 / 10) ** 2)

    def test_accountant(self):
        acc = BudgetAccountant(1.5, 0)

        a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T
        histogramdd(a, epsilon=1, bins=3, range=[(0, 10), (0, 10)], density=True, accountant=acc)

        with self.assertRaises(BudgetError):
            histogramdd(a, epsilon=1, bins=3, range=[(0, 10), (0, 10)], density=True, accountant=acc)

    def test_default_accountant(self):
        BudgetAccountant.pop_default()

        a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T
        histogramdd(a, epsilon=1, bins=3, range=[(0, 10), (0, 10)], density=True)
        acc = BudgetAccountant.pop_default()
        self.assertEqual((1, 0), acc.total())
        self.assertEqual(acc.epsilon, float("inf"))
        self.assertEqual(acc.delta, 1.0)

        histogramdd(a, epsilon=1, bins=3, range=[(0, 10), (0, 10)])

        self.assertEqual((1, 0), acc.total())
