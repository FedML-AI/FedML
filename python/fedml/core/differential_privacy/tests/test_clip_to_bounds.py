from unittest import TestCase
import numpy as np
import pytest

from fedml.core.differential_privacy.validation import clip_to_bounds


class TestClipToBounds(TestCase):
    def test_incorrect_parameterisation(self):
        with self.assertRaises(TypeError):
            clip_to_bounds([1, 2, 3], (0, 5))

        with self.assertRaises(TypeError):
            clip_to_bounds(np.ones((5, 1)), [1, 2])

        with self.assertRaises(Exception):
            clip_to_bounds(np.ones((5, 1)), ("One", "Two"))

    def test_simple(self):
        X = np.ones(shape=(5, 1))
        X2 = clip_to_bounds(X, (0, 0.5))
        self.assertTrue(np.all(X2 == X/2))

        X3 = clip_to_bounds(X, (0, 2))
        self.assertTrue(np.all(X3 == X))

        X4 = X.copy()
        X4[0, 0] = 2
        X5 = clip_to_bounds(X4, (0, 1))
        self.assertTrue(np.all(X5 == X))

    def test_bad_bounds(self):
        X = np.ones(shape=(5, 1))
        self.assertRaises(ValueError, clip_to_bounds, X, ([1], [2, 3]))

    def test_1d_array(self):
        X = np.ones(shape=(5,))
        X2 = clip_to_bounds(X, (0, 0.5))
        self.assertTrue(np.all(X2 == X/2))

        X3 = clip_to_bounds(X, (0, 2))
        self.assertTrue(np.all(X3 == X))

        X4 = X.copy()
        X4[0] = 2
        X5 = clip_to_bounds(X4, (0, 1))
        self.assertTrue(np.all(X5 == X))

    def test_3d_array(self):
        X = np.ones(shape=(2, 2, 2))
        self.assertRaises(ValueError, clip_to_bounds, X, ([0, 0.5], [0, 0.5]))

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_iris(self):
        from sklearn import datasets
        dataset = datasets.load_iris()

        X_train, y_train = dataset.data, dataset.target

        maxes = np.max(X_train, axis=1)
        clip_max = (maxes[0] + maxes[1]) / 2

        X_clipped = clip_to_bounds(X_train, (np.min(X_train), clip_max))
        clipped_maxes = np.max(X_clipped, axis=0)
        self.assertLessEqual(clipped_maxes[0], maxes[0])
        self.assertLessEqual(clipped_maxes[1], maxes[1])
        self.assertTrue(np.isclose(clipped_maxes[0], clip_max) or np.isclose(clipped_maxes[1], clip_max))

    def test_different_bounds(self):
        X = np.ones((10, 2))

        X_clipped = clip_to_bounds(X, ([0, 0], [0.5, 1]))
        self.assertTrue(np.all(X_clipped[:, 0] == 0.5))
        self.assertTrue(np.all(X_clipped[:, 1] == 1))
