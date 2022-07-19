import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.accountant import BudgetAccountant
from fedml.core.differential_privacy.tools.histograms import histogram2d
from fedml.core.differential_privacy.utils import global_seed, PrivacyLeakWarning, BudgetError


class TestHistogram2d(TestCase):
    def test_no_params(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 7, 1, 5, 9])
        with self.assertWarns(PrivacyLeakWarning):
            res = histogram2d(x, y)
        self.assertIsNotNone(res)

    def test_no_range(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 7, 1, 5, 9])
        with self.assertWarns(PrivacyLeakWarning):
            res = histogram2d(x, y, epsilon=1)
        self.assertIsNotNone(res)

    def test_missing_range(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 7, 1, 5, 9])
        with self.assertWarns(PrivacyLeakWarning):
            res = histogram2d(x, y, epsilon=1, range=[(0, 10), None])
        self.assertIsNotNone(res)

    def test_bins_instead_of_range(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 7, 1, 5, 9])
        res = histogram2d(x, y, epsilon=1, range=None, bins=([0, 1, 10], [0, 1, 10]))
        self.assertIsNotNone(res)

    def test_custom_bins(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 7, 1, 5, 9])
        res = histogram2d(x, y, epsilon=1, bins=[0, 3, 10])
        self.assertIsNotNone(res)

    def test_same_edges(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 7, 1, 5, 9])
        _, edges_x, edges_y = np.histogram2d(x, y, bins=3, range=[(0, 10), (0, 10)])
        _, dp_edges_x, dp_edges_y = histogram2d(x, y, epsilon=1, bins=3, range=[(0, 10), (0, 10)])

        self.assertTrue((edges_x == dp_edges_x).all())
        self.assertTrue((edges_y == dp_edges_y).all())

    def test_different_result(self):
        global_seed(3141592653)
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 7, 1, 5, 9])
        hist, _, _ = np.histogram2d(x, y, bins=3, range=[(0, 10), (0, 10)])
        dp_hist, _, _ = histogram2d(x, y, epsilon=0.1, bins=3, range=[(0, 10), (0, 10)])

        # print("Non-private histogram: %s" % hist)
        # print("Private histogram: %s" % dp_hist)
        self.assertTrue((hist != dp_hist).any())

    def test_density(self):
        global_seed(3141592653)
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 7, 1, 5, 9])
        dp_hist, _, _ = histogram2d(x, y, epsilon=1, bins=3, range=[(0, 10), (0, 10)], density=True)

        # print(dp_hist.sum())

        self.assertAlmostEqual(dp_hist.sum(), 1.0 * (3 / 10) ** 2)

    def test_accountant(self):
        acc = BudgetAccountant(1.5, 0)

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 7, 1, 5, 9])
        histogram2d(x, y, epsilon=1, bins=3, range=[(0, 10), (0, 10)], density=True, accountant=acc)

        with self.assertRaises(BudgetError):
            histogram2d(x, y, epsilon=1, bins=3, range=[(0, 10), (0, 10)], density=True, accountant=acc)
