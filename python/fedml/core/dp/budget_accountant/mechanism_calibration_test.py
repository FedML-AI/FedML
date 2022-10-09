"""Tests for mechanism_calibration."""

from absl.testing import absltest
from absl.testing import parameterized
import attr

import numpy as np

from fedml.core.dp.budget_accountant import dp_event
from fedml.core.dp.budget_accountant import mechanism_calibration
from fedml.core.dp.budget_accountant import privacy_accountant


@attr.define
class MockEvent(dp_event.DpEvent):
    param: float


class MockAccountant(privacy_accountant.PrivacyAccountant):

    def __init__(self, value_to_epsilon):
        super().__init__(
            privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE)
        self._value = 0.0
        self._value_to_epsilon = value_to_epsilon

    def supports(self, event: dp_event.DpEvent) -> bool:
        return True

    def _compose(self, event: dp_event.DpEvent, count: int = 1):
        self._value = event.param

    def get_epsilon(self, target_delta: float) -> float:
        return self._value_to_epsilon(self._value)


class MechanismCalibrationTest(parameterized.TestCase):

    @parameterized.parameters(
        {'eps_fn': lambda x: x, 'expected': 2.0},
        {'eps_fn': lambda x: 4 - x, 'expected': 2.0},
        {'eps_fn': np.square, 'expected': np.sqrt(2)},
        {'eps_fn': np.cbrt, 'expected': 8.0},
        {'eps_fn': lambda x: (x - 5) ** 3 + 2, 'expected': 5},
        {'eps_fn': lambda x: np.cos(x / 3) + 2, 'expected': 3 * np.pi / 2},
        {'eps_fn': lambda x: np.sin(x - 5) + (x + 3) / 4, 'expected': 5},
        {'eps_fn': lambda x: (13 - x) / 4 - np.sin(x - 5), 'expected': 5},
    )
    def test_basic_inversion(self, eps_fn, expected):
        value = mechanism_calibration.calibrate_dp_mechanism(
            lambda: MockAccountant(eps_fn), MockEvent, 2, 0,
            mechanism_calibration.ExplicitBracketInterval(0, 10), tol=1e-12)

        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, expected)

        accountant = MockAccountant(eps_fn)
        accountant.compose(MockEvent(value))
        epsilon = accountant.get_epsilon(0)
        self.assertLessEqual(epsilon, 2)

    @parameterized.parameters(
        {'eps_fn': lambda x: -1 if x < 0 else 1},
        {'eps_fn': lambda x: 1 if x < 0 else -1},
        {'eps_fn': lambda x: x - 1 if x < 0 else x + 1},
        {'eps_fn': lambda x: -2 - x if x < 0 else 2 - x},
        {'eps_fn': lambda x: x + 2 if x < 0 else x - 2},
        {'eps_fn': lambda x: 1 - x if x < 0 else -1 - x},
    )
    def test_discontinuous(self, eps_fn):
        value = mechanism_calibration.calibrate_dp_mechanism(
            lambda: MockAccountant(eps_fn), MockEvent, 0, 0,
            mechanism_calibration.ExplicitBracketInterval(-1, 1), tol=1e-12)

        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, 0)

        accountant = MockAccountant(eps_fn)
        accountant.compose(MockEvent(value))
        epsilon = accountant.get_epsilon(0)
        self.assertLessEqual(epsilon, 0)

    @parameterized.parameters(
        {'eps_fn': lambda x: x - 2, 'expected_eps': 0},
        {'eps_fn': lambda x: x - 2.1, 'expected_eps': -0.1},
        {'eps_fn': lambda x: x - 2.9, 'expected_eps': -0.9},
        {'eps_fn': lambda x: 2 - x, 'expected_eps': 0},
        {'eps_fn': lambda x: 1.9 - x, 'expected_eps': -0.1},
        {'eps_fn': lambda x: 1.1 - x, 'expected_eps': -0.9},
    )
    def test_discrete(self, eps_fn, expected_eps):
        value = mechanism_calibration.calibrate_dp_mechanism(
            lambda: MockAccountant(eps_fn), MockEvent, 0, 0,
            mechanism_calibration.ExplicitBracketInterval(0, 5), discrete=True)

        self.assertIsInstance(value, int)
        self.assertEqual(value, 2)

        accountant = MockAccountant(eps_fn)
        accountant.compose(MockEvent(value))
        epsilon = accountant.get_epsilon(0)
        self.assertAlmostEqual(epsilon, expected_eps)

    @parameterized.parameters(
        {'epsilon_gap': lambda x: x, 'lower': -1, 'guess': -0.5},
        {'epsilon_gap': lambda x: -x, 'lower': -1, 'guess': -0.5},
        {'epsilon_gap': lambda x: np.exp(x) - 2, 'lower': 0, 'guess': 0.1},
        {'epsilon_gap': lambda x: 1 - np.sqrt(x), 'lower': 0, 'guess': 0.1},
        {'epsilon_gap': lambda x: np.log(x) - 20, 'lower': 1, 'guess': 2},
    )
    def test_search_for_explicit_bracket_interval(
            self, epsilon_gap, lower, guess):
        lower_value = epsilon_gap(lower)
        interval = mechanism_calibration._search_for_explicit_bracket_interval(
            mechanism_calibration.LowerEndpointAndGuess(lower, guess), epsilon_gap)
        upper_value = epsilon_gap(interval.endpoint_2)
        self.assertLessEqual(lower_value * upper_value, 0)

    def test_raises_unknown_bracket_interval_type(self):
        class UnknownBracketInterval(mechanism_calibration.BracketInterval):
            pass

        with self.assertRaisesRegex(TypeError, 'Unrecognized bracket_interval'):
            mechanism_calibration.calibrate_dp_mechanism(
                lambda: MockAccountant(lambda x: x), MockEvent, 1.0, 0,
                UnknownBracketInterval())

    def test_raises_mfa_not_callable(self):
        with self.assertRaisesRegex(TypeError, 'callable'):
            mechanism_calibration.calibrate_dp_mechanism(
                'not a callable', MockEvent, 1.0, 0,
                mechanism_calibration.ExplicitBracketInterval(0, 5))

    def test_raises_mefv_not_callable(self):
        with self.assertRaisesRegex(TypeError, 'callable'):
            mechanism_calibration.calibrate_dp_mechanism(
                lambda: MockAccountant(lambda x: x), 'not a callable', 1.0, 0,
                mechanism_calibration.ExplicitBracketInterval(0, 5))

    def test_raises_target_epsilon_negative(self):
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            mechanism_calibration.calibrate_dp_mechanism(
                lambda: MockAccountant(lambda x: x), MockEvent, -1.0, 0,
                mechanism_calibration.ExplicitBracketInterval(0, 5))

    def test_raises_target_delta_out_of_range(self):
        with self.assertRaisesRegex(ValueError, 'in range'):
            mechanism_calibration.calibrate_dp_mechanism(
                lambda: MockAccountant(lambda x: x), MockEvent, 0.0, -0.1,
                mechanism_calibration.ExplicitBracketInterval(0, 5))

        with self.assertRaisesRegex(ValueError, 'in range'):
            mechanism_calibration.calibrate_dp_mechanism(
                lambda: MockAccountant(lambda x: x), MockEvent, 0.0, 1.1,
                mechanism_calibration.ExplicitBracketInterval(0, 5))

    def test_bad_bracket_interval(self):
        with self.assertRaisesRegex(ValueError, 'Bracket endpoints'):
            mechanism_calibration.calibrate_dp_mechanism(
                lambda: MockAccountant(lambda x: x), MockEvent, 1.0, 0.0,
                mechanism_calibration.ExplicitBracketInterval(2, 5))

        with self.assertRaisesRegex(ValueError, 'Bracket endpoints'):
            mechanism_calibration.calibrate_dp_mechanism(
                lambda: MockAccountant(lambda x: x), MockEvent, 1.0, 0.0,
                mechanism_calibration.ExplicitBracketInterval(-2, 0))

        with self.assertRaisesRegex(ValueError, 'must be less than'):
            mechanism_calibration.calibrate_dp_mechanism(
                lambda: MockAccountant(lambda x: x), MockEvent, 1.0, 0.0,
                mechanism_calibration.LowerEndpointAndGuess(2, 0))

    def test_negative_tol(self):
        with self.assertRaisesRegex(ValueError, 'tol'):
            mechanism_calibration.calibrate_dp_mechanism(
                lambda: MockAccountant(lambda x: x), MockEvent, 1.0, 0.0,
                mechanism_calibration.LowerEndpointAndGuess(0, 1), tol=-1)

    def test_no_bracket_interval_found(self):
        with self.assertRaises(mechanism_calibration.NoBracketIntervalFoundError):
            mechanism_calibration.calibrate_dp_mechanism(
                lambda: MockAccountant(lambda x: x), MockEvent, 1.0e10, 0.0,
                mechanism_calibration.LowerEndpointAndGuess(0, 1))

    def test_nonempty_accountant(self):
        def make_fresh_accountant():
            accountant = MockAccountant(lambda x: x)
            accountant.compose(MockEvent(1.0))
            return accountant

        with self.assertRaises(mechanism_calibration.NonEmptyAccountantError):
            mechanism_calibration.calibrate_dp_mechanism(
                make_fresh_accountant, MockEvent, 0.5, 0.0,
                mechanism_calibration.ExplicitBracketInterval(0, 1))


if __name__ == '__main__':
    absltest.main()
