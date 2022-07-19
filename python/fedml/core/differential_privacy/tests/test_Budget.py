from unittest import TestCase

from fedml.core.differential_privacy.utils import Budget


class TestBudget(TestCase):
    def test_tuple(self):
        self.assertIsInstance(Budget(0, 0), tuple)

    def test_correct_vals(self):
        self.assertIsNotNone(Budget(0, 0))
        self.assertIsNotNone(Budget(1, 0))
        self.assertIsNotNone(Budget(float("inf"), 1))
        self.assertIsNotNone(Budget(0, 0.1))
        self.assertIsNotNone(Budget(0, 1e-300))
        self.assertIsNotNone(Budget(1e300, 1e-300))

    def test_bad_vals(self):
        with self.assertRaises(ValueError):
            Budget(-1, 0)

        with self.assertRaises(ValueError):
            Budget(0, -0.1)

        with self.assertRaises(ValueError):
            Budget(0, 1.1)

        with self.assertRaises(TypeError):
            Budget("0", 0)

        with self.assertRaises(TypeError):
            Budget(complex(0, 1), 0)

    def test_comparisons(self):
        self.assertTrue(Budget(2, 0.2) >= Budget(2, 0.2))
        self.assertFalse(Budget(4, 0.02) >= Budget(3, 0.1))
        self.assertTrue((4, 0.2) >= (3, 0.1))

        self.assertFalse(Budget(3, 0) > (2, 0.1))
        self.assertFalse(Budget(1, 0.2) < (2, 0))
        self.assertFalse((3, 0) > Budget(2, 0.1))
        self.assertFalse((1, 0.2) < Budget(2, 0))

        # Equal
        lhs = Budget(1, 0.5)
        rhs = Budget(1, 0.5)
        self.assertTrue(lhs == rhs)
        self.assertTrue(lhs >= rhs)
        self.assertTrue(lhs <= rhs)
        self.assertFalse(lhs > rhs)
        self.assertFalse(lhs < rhs)

        # Gt
        for lhs, rhs in [(Budget(1.1, 0.75), Budget(1, 0.5)), (Budget(1.1, 0.5), Budget(1, 0.5)),
                         (Budget(1, 0.75), Budget(1, 0.5))]:
            self.assertFalse(lhs == rhs)
            self.assertTrue(lhs >= rhs)
            self.assertFalse(lhs <= rhs)
            self.assertTrue(lhs > rhs)
            self.assertFalse(lhs < rhs)

        # Ge
        for lhs, rhs in [(Budget(1.1, 0.75), Budget(1, 0.5)), (Budget(1, 0.5), Budget(1, 0.5)),
                         (Budget(1.1, 0.5), Budget(1, 0.5)), (Budget(1, 0.75), Budget(1, 0.5))]:
            self.assertTrue(lhs >= rhs)
            self.assertFalse(lhs < rhs)

        # Lt
        for lhs, rhs in [(Budget(0.9, 0.25), Budget(1, 0.5)), (Budget(0.9, 0.5), Budget(1, 0.5)),
                         (Budget(1, 0.25), Budget(1, 0.5))]:
            self.assertFalse(lhs == rhs)
            self.assertFalse(lhs >= rhs)
            self.assertTrue(lhs <= rhs)
            self.assertFalse(lhs > rhs)
            self.assertTrue(lhs < rhs)

        # Le
        for lhs, rhs in [(Budget(0.9, 0.25), Budget(1, 0.5)), (Budget(1, 0.5), Budget(1, 0.5)),
                         (Budget(0.9, 0.5), Budget(1, 0.5)), (Budget(1, 0.25), Budget(1, 0.5))]:
            self.assertFalse(lhs > rhs)
            self.assertTrue(lhs <= rhs)

        # inf
        self.assertFalse(Budget(2, 0.1) > Budget(float("inf"), 1))

    def test_repr(self):
        self.assertIn("epsilon=", repr(Budget(0, 0)))
        self.assertIn("delta=", repr(Budget(0, 0)))
