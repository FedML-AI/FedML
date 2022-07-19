from unittest import TestCase

from fedml.core.differential_privacy.validation import check_epsilon_delta


class TestCheckEpsilonDelta(TestCase):
    def test_real_inputs(self):
        with self.assertRaises(TypeError):
            check_epsilon_delta("1", "0")

        with self.assertRaises(TypeError):
            check_epsilon_delta(complex(1, 0.5), 0)

        with self.assertRaises(TypeError):
            check_epsilon_delta([1], [0])

        self.assertIsNone(check_epsilon_delta(1, 0))
        self.assertIsNone(check_epsilon_delta(1.0, 0.0))

    def test_all_zero(self):
        with self.assertRaises(ValueError):
            check_epsilon_delta(0, 0)

        with self.assertRaises(ValueError):
            check_epsilon_delta(0.0, 0.0)

    def test_neg_eps(self):
        with self.assertRaises(ValueError):
            check_epsilon_delta(-1, 0)

        with self.assertRaises(ValueError):
            check_epsilon_delta(-1e-100, 0)

    def test_wrong_delta(self):
        with self.assertRaises(ValueError):
            check_epsilon_delta(0, -1)

        with self.assertRaises(ValueError):
            check_epsilon_delta(0, 1.1)

    def test_max_eps_delt(self):
        self.assertIsNone(check_epsilon_delta(float("inf"), 1))
