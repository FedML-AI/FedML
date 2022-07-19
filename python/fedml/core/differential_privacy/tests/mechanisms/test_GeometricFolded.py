import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import GeometricFolded
from fedml.core.differential_privacy.utils import global_seed


class TestGeometricFolded(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = GeometricFolded

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMechanism
        self.assertTrue(issubclass(GeometricFolded, DPMechanism))

    def test_default_sensitivity(self):
        mech = self.mech(epsilon=1, lower=0, upper=10)

        self.assertEqual(1, mech.sensitivity)
        self.assertIsNotNone(mech.randomise(1))

    def test_non_integer_sensitivity(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, sensitivity=0.5, lower=0, upper=10)

    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1, sensitivity=0, lower=0, upper=2)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_non_zero_delta(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=10)
        mech.delta = 0.5

        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=-1, sensitivity=1, lower=0, upper=10)

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), sensitivity=1, lower=0, upper=10)

        for i in range(1000):
            self.assertEqual(mech.randomise(1), 1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1 + 2j, sensitivity=1, lower=0, upper=10)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon="Two", sensitivity=1, lower=0, upper=10)

    def test_half_integer_bounds(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=1.5)
        val = mech.randomise(0)
        self.assertIsInstance(val, int)

    def test_non_half_integer_bounds(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1, sensitivity=1, lower=0, upper=2.2)

    def test_inf_bounds(self):
        self.assertIsNotNone(self.mech(epsilon=1, lower=0, upper=float("inf")))
        self.assertIsNotNone(self.mech(epsilon=1, lower=-float("inf"), upper=0))
        self.assertIsNotNone(self.mech(epsilon=1, lower=-float("inf"), upper=float("inf")))

    def test_non_numeric(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=10)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_non_integer(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=10)
        with self.assertRaises(TypeError):
            mech.randomise(1.0)

    def test_zero_median_prob(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=4)
        vals = []

        for i in range(1000):
            vals.append(mech.randomise(2))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 2.0, delta=0.1)

    def test_neighbors_prob(self):
        epsilon = np.log(2)
        runs = 1000
        mech = self.mech(epsilon=epsilon, sensitivity=1, lower=0, upper=4)
        count = [0, 0]

        for i in range(runs):
            val0 = mech.randomise(1)
            if val0 <= 1:
                count[0] += 1

            val1 = mech.randomise(2)
            if val1 <= 1:
                count[1] += 1

        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0], np.exp(epsilon) * count[1] + 0.15 * runs)

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, sensitivity=1, lower=0, upper=4))
        self.assertIn(".GeometricFolded(", repr_)

    def test_bias(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, sensitivity=1, lower=0, upper=1).bias, 0)

    def test_variance(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, sensitivity=1, lower=0, upper=1).variance, 0)
