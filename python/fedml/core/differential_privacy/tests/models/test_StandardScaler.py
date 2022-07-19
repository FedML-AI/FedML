from unittest import TestCase

import numpy as np
import sklearn.preprocessing as sk_pp

from fedml.core.differential_privacy.models.standard_scaler import StandardScaler
from fedml.core.differential_privacy.utils import PrivacyLeakWarning, DiffprivlibCompatibilityWarning, global_seed, BudgetError


class TestStandardScaler(TestCase):
    def test_class(self):
        from sklearn.base import BaseEstimator
        self.assertTrue(issubclass(StandardScaler, BaseEstimator))

    def test_not_none(self):
        ss = StandardScaler()
        self.assertIsNotNone(ss)

    def test_no_range(self):
        X = np.random.rand(10, 5)
        ss = StandardScaler()

        with self.assertWarns(PrivacyLeakWarning):
            ss.fit(X)

    def test_do_nothing(self):
        X = np.random.rand(10, 5)

        dp_ss = StandardScaler(bounds=(0, 1), epsilon=1, with_std=False, with_mean=False)
        dp_ss.fit(X)

        self.assertIsNotNone(dp_ss)

    def test_refit(self):
        X = np.random.rand(10, 5)

        dp_ss = StandardScaler(bounds=(0, 1), epsilon=1)
        dp_ss.fit(X)
        dp_ss.partial_fit(X)
        self.assertIsNotNone(dp_ss)

    def test_sample_weights(self):
        X = np.random.rand(10, 5)

        dp_ss = StandardScaler(bounds=(0, 1), epsilon=1)

        try:
            self.assertWarns(DiffprivlibCompatibilityWarning, dp_ss.fit, X, sample_weight=1)
        except TypeError:
            pass

    def test_inf_epsilon(self):
        X = np.random.rand(10, 5)

        dp_ss = StandardScaler(bounds=(0, 1), epsilon=float("inf"))
        dp_ss.fit(X)

        sk_ss = sk_pp.StandardScaler()
        sk_ss.fit(X)

        self.assertTrue(np.allclose(dp_ss.mean_, sk_ss.mean_), "Arrays %s and %s should be the same" %
                        (dp_ss.mean_, sk_ss.mean_))
        self.assertTrue(np.allclose(dp_ss.var_, sk_ss.var_), "Arrays %s and %s should be the same" %
                        (dp_ss.var_, sk_ss.var_))
        self.assertTrue(np.all(dp_ss.n_samples_seen_ == sk_ss.n_samples_seen_))

    def test_different_results(self):
        X = np.random.rand(10, 5)

        ss1 = StandardScaler(bounds=(0, 1))
        ss1.fit(X)

        ss2 = StandardScaler(bounds=(0, 1))
        ss2.fit(X)

        self.assertFalse(np.allclose(ss1.mean_, ss2.mean_), "Arrays %s and %s should be different" %
                         (ss1.mean_, ss2.mean_))
        self.assertFalse(np.allclose(ss1.var_, ss2.var_), "Arrays %s and %s should be different" %
                         (ss1.var_, ss2.var_))

    def test_functionality(self):
        X = np.random.rand(10, 5)
        ss = StandardScaler(bounds=(0, 1))
        ss.fit(X)

        self.assertIsNotNone(ss.transform(X))
        self.assertIsNotNone(ss.inverse_transform(X))
        self.assertIsNotNone(ss.fit_transform(X))

    def test_similar_results(self):
        global_seed(314159)

        X = np.random.rand(100000, 5)

        dp_ss = StandardScaler(bounds=(0, 1), epsilon=float("inf"))
        dp_ss.fit(X)

        sk_ss = sk_pp.StandardScaler()
        sk_ss.fit(X)

        self.assertTrue(np.allclose(dp_ss.mean_, sk_ss.mean_, rtol=1, atol=1e-4), "Arrays %s and %s should be close" %
                        (dp_ss.mean_, sk_ss.mean_))
        self.assertTrue(np.allclose(dp_ss.var_, sk_ss.var_, rtol=1, atol=1e-4), "Arrays %s and %s should be close" %
                        (dp_ss.var_, sk_ss.var_))
        self.assertTrue(np.all(dp_ss.n_samples_seen_ == sk_ss.n_samples_seen_))

    def test_accountant(self):
        from fedml.core.differential_privacy.accountant import BudgetAccountant
        acc = BudgetAccountant()

        X = np.random.rand(10, 5)
        ss = StandardScaler(epsilon=1, bounds=(0, 1), accountant=acc)
        ss.fit(X)
        self.assertEqual((1, 0), acc.total())

        with BudgetAccountant(1.5, 0) as acc2:
            ss = StandardScaler(epsilon=1, bounds=(0, 1))
            ss.fit(X)
            self.assertEqual((1, 0), acc2.total())

            with self.assertRaises(BudgetError):
                ss.fit(X)

        self.assertEqual((1, 0), acc.total())
