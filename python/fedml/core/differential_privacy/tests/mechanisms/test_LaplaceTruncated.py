import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import LaplaceTruncated
from fedml.core.differential_privacy.utils import global_seed


class TestLaplaceTruncated(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = LaplaceTruncated

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMechanism
        self.assertTrue(issubclass(LaplaceTruncated, DPMechanism))

    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=0, lower=0, upper=2)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), delta=0, sensitivity=1, lower=0, upper=1)

        for i in range(1000):
            self.assertEqual(mech.randomise(0.5), 0.5)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1 + 2j, delta=0, sensitivity=1, lower=0, upper=1)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon="Two", delta=0, sensitivity=1, lower=0, upper=1)

    def test_wrong_bounds(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1, delta=0, sensitivity=1, lower=3, upper=1)

        with self.assertRaises(TypeError):
            self.mech(epsilon=1, delta=0, sensitivity=1, lower="0", upper="2")

    def test_non_numeric(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_zero_median_prob(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(0.5))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.5, delta=0.1)

    def test_neighbors_prob(self):
        epsilon = 1
        runs = 10000
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        count = [0, 0]

        for i in range(runs):
            val0 = mech.randomise(0)
            if val0 <= 0.5:
                count[0] += 1

            val1 = mech.randomise(1)
            if val1 <= 0.5:
                count[1] += 1

        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.1)

    def test_within_bounds(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        vals = []

        for i in range(1000):
            vals.append(mech.randomise(0.5))

        vals = np.array(vals)

        self.assertTrue(np.all(vals >= 0))
        self.assertTrue(np.all(vals <= 1))

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1))
        self.assertIn(".LaplaceTruncated(", repr_)

    def test_bias(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        self.assertGreater(mech.bias(0), 0.0)
        self.assertLess(mech.bias(1), 0.0)

    def test_variance(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        self.assertGreater(mech.variance(0), 0.0)
