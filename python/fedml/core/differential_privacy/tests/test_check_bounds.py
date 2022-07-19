from numbers import Real
from unittest import TestCase

import numpy as np

from fedml.core.differential_privacy.validation import check_bounds


class TestCheckBounds(TestCase):
    def test_none(self):
        self.assertRaises(TypeError, check_bounds, None)

    def test_non_tuple(self):
        with self.assertRaises(TypeError):
            check_bounds([1, 2, 3])

    def test_incorrect_entries(self):
        with self.assertRaises(ValueError):
            check_bounds(([1, 2], 1))

        with self.assertRaises(ValueError):
            check_bounds(([1, 2], [1, 2, 3]))

        with self.assertRaises(ValueError):
            check_bounds(([1, 2], [1, 2], [1, 2]))

    def test_consistency(self):
        bounds = check_bounds(([1, 1], [2, 2]), shape=2)
        bounds2 = check_bounds(bounds, shape=2)
        self.assertTrue(np.all(bounds[0] == bounds2[0]))
        self.assertTrue(np.all(bounds[1] == bounds2[1]))

    def test_array_output(self):
        bounds = check_bounds(([1, 1], [2, 2]), shape=2)
        self.assertIsInstance(bounds[0], np.ndarray)
        self.assertIsInstance(bounds[1], np.ndarray)

    def test_scalar_output(self):
        bounds = check_bounds((1, 2), shape=0)
        self.assertIsInstance(bounds[0], Real)
        self.assertIsInstance(bounds[1], Real)

        bounds = check_bounds((1, 2), shape=0, dtype=int)
        self.assertIsInstance(bounds[0], int)
        self.assertIsInstance(bounds[1], int)

        bounds = check_bounds((1, 2), shape=0, dtype=float)
        self.assertIsInstance(bounds[0], float)
        self.assertIsInstance(bounds[1], float)

    def test_wrong_dims(self):
        with self.assertRaises(ValueError):
            check_bounds(([1, 1], [2, 2]), shape=3)

        with self.assertRaises(ValueError):
            check_bounds(([[1, 1]], [[2, 2]]), shape=2)

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            check_bounds(([1, 1], [2, 2]), shape=-2)

        with self.assertRaises(TypeError):
            check_bounds(([1, 1], [2, 2]), shape=2.0)

    def test_wrong_order(self):
        with self.assertRaises(ValueError):
            check_bounds((2, 1))

    def test_non_numeric(self):
        with self.assertRaises(ValueError):
            check_bounds(("One", "Two"))

    def test_complex(self):
        with self.assertRaises(TypeError):
            check_bounds((1.0, 1+2j), dtype=complex)

    def test_min_separation(self):
        bounds = check_bounds((1, 1), min_separation=2)
        self.assertEqual(0, bounds[0])
        self.assertEqual(2, bounds[1])

        bounds = check_bounds((1., 1.), min_separation=1)
        self.assertEqual(0.5, bounds[0])
        self.assertEqual(1.5, bounds[1])

        bounds = check_bounds((0.9, 1.1), min_separation=1)
        self.assertEqual(0.5, bounds[0])
        self.assertEqual(1.5, bounds[1])
