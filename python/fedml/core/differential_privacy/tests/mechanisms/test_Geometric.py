import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import Geometric
from fedml.core.differential_privacy.utils import global_seed


class TestGeometric(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Geometric

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMechanism
        self.assertTrue(issubclass(Geometric, DPMechanism))

    def test_default_sensitivity(self):
        mech = self.mech(epsilon=1)

        self.assertEqual(1, mech.sensitivity)
        self.assertIsNotNone(mech.randomise(1))

    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1, sensitivity=0)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_neg_sensitivity(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1, sensitivity=-1)

    def test_non_integer_sensitivity(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, sensitivity=0.5)

    def test_non_zero_delta(self):
        mech = self.mech(epsilon=1, sensitivity=1)
        mech.delta = 0.5

        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=-1, sensitivity=1)

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), sensitivity=1)

        for i in range(1000):
            self.assertEqual(mech.randomise(1), 1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1 + 2j, sensitivity=1)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon="Two", sensitivity=1)

    def test_non_numeric(self):
        mech = self.mech(epsilon=1, sensitivity=1)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_non_integer(self):
        mech = self.mech(epsilon=1, sensitivity=1)
        with self.assertRaises(TypeError):
            mech.randomise(1.0)

    def test_zero_median(self):
        mech = self.mech(epsilon=1, sensitivity=1)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)

    def test_neighbors_prob(self):
        epsilon = 1
        runs = 10000
        mech = self.mech(epsilon=epsilon, sensitivity=1)
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

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, sensitivity=1))
        self.assertIn(".Geometric(", repr_)

    def test_bias(self):
        self.assertEqual(0.0, self.mech(epsilon=1, sensitivity=1).bias(0))

    def test_variance(self):
        mech = self.mech(epsilon=-np.log(0.5))

        # Expected answer gives \sum_{i \in Z} i^2 (1/2)^i  = 2 \sum_{i > 0} i^2 (1/2)^i
        self.assertAlmostEqual(6 * 2 / 3, mech.variance(0))
