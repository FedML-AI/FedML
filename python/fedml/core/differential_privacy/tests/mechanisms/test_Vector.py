import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import Vector
from fedml.core.differential_privacy.utils import global_seed


class TestVector(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Vector

    def teardown_method(self, method):
        del self.mech

    @staticmethod
    def func(x):
        return np.sum(x ** 2)

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMechanism
        self.assertTrue(issubclass(Vector, DPMechanism))

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=-1, dimension=3, function_sensitivity=1)

    def test_nonzero_delta(self):
        mech = self.mech(epsilon=1, dimension=3, function_sensitivity=1)
        mech.delta = 0.1

        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), dimension=3, function_sensitivity=1)

        for i in range(100):
            noisy_func = mech.randomise(self.func)
            self.assertAlmostEqual(noisy_func(np.zeros(3)), 0)
            self.assertAlmostEqual(noisy_func(np.ones(3)), 3)

    def test_wrong_sensitivity(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, dimension=3, function_sensitivity="1")

        with self.assertRaises(TypeError):
            self.mech(epsilon=1, dimension=3, function_sensitivity=1, data_sensitivity="1")

        with self.assertRaises(ValueError):
            self.mech(epsilon=1, dimension=3, function_sensitivity=-1)

        with self.assertRaises(ValueError):
            self.mech(epsilon=1, dimension=3, function_sensitivity=1, data_sensitivity=-1)

    def test_zero_data_sensitivity(self):
        mech = self.mech(epsilon=1, dimension=3, function_sensitivity=1, data_sensitivity=0)

        for i in range(100):
            noisy_func = mech.randomise(self.func)
            self.assertAlmostEqual(noisy_func(np.zeros(3)), 0)
            self.assertAlmostEqual(noisy_func(np.ones(3)), 3)

    def test_wrong_alpha(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, dimension=3, function_sensitivity=1, alpha="1")

        with self.assertRaises(ValueError):
            self.mech(epsilon=1, dimension=3, function_sensitivity=1, alpha=-1)

    def test_wrong_dimension(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, dimension=1.2, function_sensitivity=1)

        with self.assertRaises(ValueError):
            self.mech(epsilon=1, dimension=0, function_sensitivity=1)

    def test_numeric_input(self):
        mech = self.mech(epsilon=1, dimension=3, function_sensitivity=1)

        with self.assertRaises(TypeError):
            mech.randomise(1)

    def test_string_input(self):
        mech = self.mech(epsilon=1, dimension=3, function_sensitivity=1)

        with self.assertRaises(TypeError):
            mech.randomise("1")

    def test_sets_once(self):
        mech = self.mech(epsilon=1, dimension=3, function_sensitivity=1)
        noisy_func = mech.randomise(self.func)
        answer = noisy_func(np.ones(3))

        for i in range(10):
            self.assertEqual(noisy_func(np.ones(3)), answer)

    def test_different_result(self):
        mech = self.mech(epsilon=1, dimension=3, function_sensitivity=1)
        noisy_func = mech.randomise(self.func)

        for i in range(10):
            old_noisy_func = noisy_func
            noisy_func = mech.randomise(self.func)

            self.assertNotAlmostEqual(noisy_func(np.ones(3)), 3)
            self.assertNotAlmostEqual(noisy_func(np.ones(3)), old_noisy_func(np.ones(3)))
            # print(noisy_func(np.ones(3)))

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, dimension=4, function_sensitivity=1))
        self.assertIn(".Vector(", repr_)

    def test_bias(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, dimension=4, function_sensitivity=1).bias, 0)

    def test_variance(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, dimension=4, function_sensitivity=1).variance, 0)
