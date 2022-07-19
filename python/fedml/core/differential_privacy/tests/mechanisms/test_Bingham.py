import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import Bingham
from fedml.core.differential_privacy.utils import global_seed


class TestBingham(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Bingham
        a = np.random.random((5, 3))
        self.random_array = a.T.dot(a)

    def teardown_method(self, method):
        del self.mech

    @staticmethod
    def generate_data(d=5, n=10):
        a = np.random.random((n, d))
        return a.T.dot(a)

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMechanism
        self.assertTrue(issubclass(Bingham, DPMechanism))

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=-1)

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"))

        for i in range(100):
            data = self.generate_data()
            eigvals, eigvecs = np.linalg.eigh(data)
            true_data = eigvecs[:, eigvals.argmax()]

            noisy_data = mech.randomise(data)
            self.assertTrue(np.allclose(true_data, noisy_data))

    def test_non_zero_delta(self):
        mech = self.mech(epsilon=1)
        mech.delta = 0.5

        with self.assertRaises(ValueError):
            mech.randomise(self.generate_data())

    def test_default_sensitivity(self):
        mech = self.mech(epsilon=1)
        self.assertEqual(mech.sensitivity, 1.0)

    def test_wrong_sensitivity(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, sensitivity="1")

        with self.assertRaises(ValueError):
            self.mech(epsilon=1, sensitivity=-1)

    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1, sensitivity=0)

        for i in range(100):
            data = self.generate_data()
            eigvals, eigvecs = np.linalg.eigh(data)
            true_data = eigvecs[:, eigvals.argmax()]

            noisy_data = mech.randomise(data)
            self.assertTrue(np.allclose(true_data, noisy_data))

    def test_numeric_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(TypeError):
            mech.randomise(1)

    def test_string_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(TypeError):
            mech.randomise("1")

    def test_list_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(TypeError):
            mech.randomise([1, 2, 3])

    def test_string_array_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(TypeError):
            mech.randomise(np.array([["1", "2"], ["3", "4"]]))

    def test_scalar_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(ValueError):
            mech.randomise(np.array([1]))

    def test_scalar_array_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        self.assertIsNotNone(mech.randomise(np.array([[1]])))

    def test_vector_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(ValueError):
            mech.randomise(np.array([1, 2, 3]))

    def test_non_square_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(ValueError):
            mech.randomise(np.ones((3, 4)))

    def test_non_symmetric_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)
        data = self.generate_data()
        data[0, 1] -= 1

        with self.assertRaises(ValueError):
            mech.randomise(data)

    def test_3D_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(ValueError):
            mech.randomise(np.ones((3, 3, 3)))

    def test_large_input(self):
        X = np.random.randn(10000, 21)
        X -= np.mean(X, axis=0)
        X /= np.linalg.norm(X, axis=1).max()
        XtX = X.T.dot(X)

        mech = self.mech(epsilon=1)
        self.assertIsNotNone(mech.randomise(XtX))

    def test_different_result(self):
        mech = self.mech(epsilon=1, sensitivity=1)
        data = self.generate_data()
        noisy_data = mech.randomise(data)

        for i in range(10):
            old_noisy_data = noisy_data
            noisy_data = mech.randomise(self.generate_data())

            self.assertTrue(np.isclose(noisy_data.dot(noisy_data), 1.0))
            self.assertFalse(np.allclose(noisy_data, old_noisy_data))

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, sensitivity=1))
        self.assertIn(".Bingham(", repr_)

    def test_bias(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, sensitivity=1).bias, np.array([[1]]))

    def test_variance(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, sensitivity=1).variance, np.array([[1]]))
