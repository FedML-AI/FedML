from unittest import TestCase

import numpy as np

from fedml.core.differential_privacy.models.utils import covariance_eig
from fedml.core.differential_privacy.utils import PrivacyLeakWarning


class TestCovarianceEig(TestCase):
    @staticmethod
    def generate_normed_data(d=5, n=10):
        a = np.random.random((n, d))
        a /= np.linalg.norm(a, axis=1).max()
        return a

    def test_simple(self):
        d = 5
        data = self.generate_normed_data(d)

        vals, vecs = covariance_eig(data, norm=1)
        self.assertIsNotNone(vals)
        self.assertIsNotNone(vecs)

        self.assertEqual(d, vals.size)
        self.assertEqual(d, vecs.shape[0])
        # Unitary matrix output
        self.assertTrue(np.allclose(vecs.dot(vecs.T), np.eye(d)))
        self.assertTrue(np.all(vals >= 0))

    def test_eigenvals_only(self):
        data = self.generate_normed_data()
        out = covariance_eig(data, norm=1, eigvals_only=True)
        self.assertNotIsInstance(out, tuple)

        out = covariance_eig(data, norm=1, eigvals_only=False)
        self.assertIsInstance(out, tuple)

    def test_large_dims(self):
        n, d = 10, 3
        data = self.generate_normed_data(d, n)
        out = covariance_eig(data, norm=1, dims=50)
        self.assertIsNotNone(out)
        self.assertEqual(out[0].size, 3)
        self.assertEqual(out[1].size, 3*3)

    def test_inf_epsilon(self):
        d, n = 3, 50
        data = self.generate_normed_data(d, n)

        dp_vals, dp_vecs = covariance_eig(data, epsilon=float("inf"), norm=1)
        vals, vecs = np.linalg.eig(data.T.dot(data))

        self.assertTrue(np.allclose(vals[vals.argsort()], dp_vals[dp_vals.argsort()]))
        self.assertTrue(np.allclose(abs(dp_vecs.T.dot(vecs).sum(axis=1)), 1))
        self.assertTrue(np.allclose(abs(dp_vecs.T.dot(vecs).sum(axis=0)), 1))

    def test_large_norm(self):
        d, n = 3, 10
        data = self.generate_normed_data(d, n)
        data *= 2

        dp_vals, dp_vecs = covariance_eig(data, epsilon=float("inf"), norm=2)
        vals, vecs = np.linalg.eigh(data.T.dot(data))

        self.assertTrue(np.allclose(vals[vals.argsort()], dp_vals[dp_vals.argsort()]))
        self.assertTrue(np.allclose(abs(dp_vecs.T.dot(vecs).sum(axis=1)), 1))
        self.assertTrue(np.allclose(abs(dp_vecs.T.dot(vecs).sum(axis=0)), 1))

    def test_bad_norm(self):
        d, n = 3, 10
        data = self.generate_normed_data(d, n)
        data *= 2

        with self.assertWarns(PrivacyLeakWarning):
            covariance_eig(data, epsilon=float("inf"), norm=None)

        with self.assertRaises(ValueError):
            covariance_eig(data, epsilon=float("inf"), norm=1)

    def test_dims(self):
        d, n = 5, 10
        data = self.generate_normed_data(d, n)

        vals, vecs = covariance_eig(data, norm=1, dims=3)
        self.assertEqual(vecs.shape, (5, 3))
        self.assertEqual(vals.shape, (5,))

        vals, vecs = covariance_eig(data, norm=1, dims=10)
        self.assertEqual(vecs.shape, (5, 5))
        self.assertEqual(vals.shape, (5,))

        vals, vecs = covariance_eig(data, norm=1, dims=0)
        self.assertEqual(vecs.shape, (5, 0))
        self.assertEqual(vals.shape, (5,))

        with self.assertRaises(ValueError):
            covariance_eig(data, dims=-5, norm=1)

        with self.assertRaises(TypeError):
            covariance_eig(data, dims=0.5, norm=1)

    def test_svd(self):
        data = self.generate_normed_data(5, 10)

        u, s, v = np.linalg.svd(data.T.dot(data))
        vals, vecs = covariance_eig(data, norm=1, epsilon=float("inf"))

        self.assertTrue(np.allclose(vals, s))
        self.assertTrue(np.allclose(abs(vecs.T.dot(u)), np.eye(5)))
        self.assertTrue(np.allclose(abs(vecs.T.dot(v.T)), np.eye(5)))
