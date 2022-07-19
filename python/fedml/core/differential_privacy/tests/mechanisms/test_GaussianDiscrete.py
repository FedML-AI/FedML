import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import GaussianDiscrete
from fedml.core.differential_privacy.utils import global_seed


class TestGaussianDiscrete(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = GaussianDiscrete

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMechanism
        self.assertTrue(issubclass(GaussianDiscrete, DPMechanism))

    def test_no_sensitivity(self):
        mech = self.mech(epsilon=1.5, delta=0.1)
        self.assertIsNotNone(mech.randomise(0))

    def test_neg_sensitivity(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1.5, delta=0.1, sensitivity=-1)

    def test_non_int_sensitivity(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1.5, delta=0.1, sensitivity=1.5)

    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1.5, delta=0.1, sensitivity=0)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_zero_epsilon_delta(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=0, delta=0)

    def test_large_epsilon(self):
        mech = self.mech(epsilon=5, delta=0.1)
        self.assertIsNotNone(mech.randomise(0))

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), delta=0.1, sensitivity=1)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1 + 2j, delta=0.1, sensitivity=1)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon="1", delta=0.1, sensitivity=1)

    def test_non_numeric(self):
        mech = self.mech(epsilon=1.5, delta=0.1, sensitivity=1)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_non_int(self):
        mech = self.mech(epsilon=1.5, delta=0.1)
        with self.assertRaises(TypeError):
            mech.randomise(1.1)

    def test_simple(self):
        mech = self.mech(epsilon=2, delta=0.1)

        self.assertIsNotNone(mech.randomise(3))

    def test_zero_median_prob(self):
        mech = self.mech(epsilon=0.5, delta=0.1)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(1))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 1, delta=0.1)

    def test_zero_median_sens_prob(self):
        mech = self.mech(epsilon=0.5, delta=0.1, sensitivity=4)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(1))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 1, delta=0.1)

    def test_neighbors_prob(self):
        epsilon = 1
        runs = 1000
        mech = self.mech(epsilon=epsilon, delta=0.1)
        count = [0, 0]

        for i in range(runs):
            val0 = mech.randomise(0)
            if val0 <= 0:
                count[0] += 1

            val1 = mech.randomise(1)
            if val1 <= 0:
                count[1] += 1

        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.1)

    def test_sanity_scale(self):
        scale1 = self.mech(epsilon=1, delta=0.5)._scale

        scale2 = self.mech(epsilon=2, delta=0.75)._scale

        self.assertGreater(scale1, scale2)

        scale3 = self.mech(epsilon=3, delta=4.00902e-10)._scale

        self.assertAlmostEqual(2, scale3, places=5)

    def test_bias(self):
        self.assertEqual(0, self.mech(epsilon=1, delta=0.5).bias(0))

    def test_variance(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, delta=0.5).variance, 0)

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, delta=0.5))
        self.assertIn(".GaussianDiscrete(", repr_)
