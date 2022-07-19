from unittest import TestCase
import numpy as np

from fedml.core.differential_privacy.mechanisms import Laplace
from fedml.core.differential_privacy.utils import global_seed


class TestLaplace(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Laplace

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMechanism
        self.assertTrue(issubclass(Laplace, DPMechanism))

    def test_neg_sensitivity(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1, delta=0, sensitivity=-1)

    def test_str_sensitivity(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, delta=0, sensitivity="1")

    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=0)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_default_delta(self):
        mech = self.mech(epsilon=1, sensitivity=1)
        self.assertEqual(0.0, mech.delta)

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=-1, delta=0, sensitivity=1)

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), delta=0, sensitivity=1)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1 + 2j, delta=0, sensitivity=1)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon="Two", delta=0, sensitivity=1)

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, delta=0, sensitivity=1))
        self.assertIn(".Laplace(", repr_)

    def test_zero_epsilon_with_delta(self):
        mech = self.mech(epsilon=0, delta=0.5, sensitivity=1)
        self.assertIsNotNone(mech.randomise(1))

    def test_epsilon_delta(self):
        mech = self.mech(epsilon=1, delta=0.01, sensitivity=1)
        self.assertIsNotNone(mech.randomise(1))

    def test_non_numeric(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_zero_median_prob(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)

    def test_neighbours_prob(self):
        epsilon = 1
        runs = 10000
        mech = self.mech(epsilon=epsilon, delta=0, sensitivity=1)
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

    def test_bias(self):
        self.assertEqual(0.0, self.mech(epsilon=1, delta=0, sensitivity=1).bias(0))

    def test_variance(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1)
        self.assertEqual(2.0, mech.variance(0))
