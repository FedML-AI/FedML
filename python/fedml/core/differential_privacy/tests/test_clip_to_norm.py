from unittest import TestCase
import numpy as np
import pytest

from fedml.core.differential_privacy.validation import clip_to_norm


class TestClipToNorm(TestCase):
    def test_incorrect_parameterisation(self):
        with self.assertRaises(TypeError):
            clip_to_norm([1, 2, 3], 1)

        with self.assertRaises(ValueError):
            clip_to_norm(np.array([1, 2, 3]), 1)

        with self.assertRaises(TypeError):
            clip_to_norm(np.ones((5, 1)), complex(1, 2))

        with self.assertRaises(TypeError):
            clip_to_norm(np.ones((5, 1)), "1")

        with self.assertRaises(ValueError):
            clip_to_norm(np.ones((5, 1)), 0)

        with self.assertRaises(ValueError):
            clip_to_norm(np.ones((5, 1)), -1)

    def test_simple(self):
        X = np.ones(shape=(5, 1))
        X2 = clip_to_norm(X, 0.5)
        self.assertTrue(np.all(X2 == X/2))

        X3 = clip_to_norm(X, 2)
        self.assertTrue(np.all(X3 == X))

        X4 = X.copy()
        X4[0, 0] = 2
        X5 = clip_to_norm(X4, 1)
        self.assertTrue(np.all(X5 == X))

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_iris(self):
        from sklearn import datasets
        dataset = datasets.load_iris()

        X_train, y_train = dataset.data, dataset.target

        norms = np.linalg.norm(X_train, axis=1)
        clip = (norms[0] + norms[1]) / 2

        X_clipped = clip_to_norm(X_train, clip)
        clipped_norms = np.linalg.norm(X_clipped, axis=1)
        self.assertLessEqual(clipped_norms[0], norms[0])
        self.assertLessEqual(clipped_norms[1], norms[1])
        self.assertTrue(np.isclose(clipped_norms[0], clip) or np.isclose(clipped_norms[1], clip))
