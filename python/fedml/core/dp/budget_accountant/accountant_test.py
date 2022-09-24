"""Tests for accountant."""

import unittest
from absl.testing import parameterized

from fedml.core.dp.budget_accountant import accountant
from fedml.core.dp.budget_accountant import common


class AccountantTest(parameterized.TestCase):

    @parameterized.named_parameters(
        {
            'testcase_name': 'basic_composition',
            'sensitivity': 21,
            'epsilon': 3,
            'delta': 0,
            'num_queries': 10,
            'expected_parameter': 70,
        },
        {
            'testcase_name': 'positive_delta',
            'sensitivity': 1,
            'epsilon': 1,
            'delta': 0.0001,
            'num_queries': 20,
            'expected_parameter': 13.6,
        },
        {
            'testcase_name': 'positive_delta_varying_sensitivity',
            'sensitivity': 0.5,
            'epsilon': 1,
            'delta': 0.0001,
            'num_queries': 20,
            'expected_parameter': 6.8,
        },
        {
            'testcase_name': 'large_num_composition',
            'sensitivity': 1,
            'epsilon': 1,
            'delta': 0.0001,
            'num_queries': 500,
            'expected_parameter': 71.2,
        }, )
    def test_get_smallest_laplace_noise(self, epsilon, delta, num_queries,
                                        sensitivity, expected_parameter):
        privacy_parameters = common.DifferentialPrivacyParameters(
            epsilon, delta)
        self.assertAlmostEqual(
            expected_parameter,
            accountant.get_smallest_laplace_noise(
                privacy_parameters, num_queries, sensitivity=sensitivity),
            delta=0.1)

    @parameterized.named_parameters(
        {
            'testcase_name': 'basic_composition',
            'sensitivity': 2,
            'epsilon': 3,
            'delta': 0,
            'num_queries': 5,
            'expected_parameter': 0.3,
        },
        {
            'testcase_name': 'positive_delta',
            'sensitivity': 1,
            'epsilon': 1,
            'delta': 0.0001,
            'num_queries': 20,
            'expected_parameter': 0.073,
        },
        {
            'testcase_name': 'positive_delta_varying_sensitivity',
            'sensitivity': 5,
            'epsilon': 1,
            'delta': 0.0001,
            'num_queries': 20,
            'expected_parameter': 0.014,
        }, )
    def test_get_smallest_discrete_laplace_noise(self, epsilon, delta,
                                                 num_queries, sensitivity,
                                                 expected_parameter):
        privacy_parameters = common.DifferentialPrivacyParameters(
            epsilon, delta)
        self.assertAlmostEqual(
            expected_parameter,
            accountant.get_smallest_discrete_laplace_noise(
                privacy_parameters, num_queries, sensitivity=sensitivity),
            delta=1e-3)

    @parameterized.named_parameters(
        {
            'testcase_name': 'base',
            'sensitivity': 1,
            'epsilon': 1,
            'delta': 0.78760074,
            'num_queries': 1,
            'expected_std': 1 / 3,
        },
        {
            'testcase_name': 'varying_sensitivity_and_num_queries',
            'sensitivity': 6,
            'epsilon': 1,
            'delta': 0.78760074,
            'num_queries': 25,
            'expected_std': 10,
        })
    def test_get_smallest_gaussian_noise(self, epsilon, delta, num_queries,
                                         sensitivity, expected_std):
        privacy_parameters = common.DifferentialPrivacyParameters(
            epsilon, delta)
        self.assertAlmostEqual(
            expected_std,
            accountant.get_smallest_gaussian_noise(
                privacy_parameters, num_queries, sensitivity=sensitivity))

    @parameterized.named_parameters(
        {
            'testcase_name': 'basic_composition',
            'epsilon': 1,
            'delta': 0,
            'total_delta': 0,
            'num_queries': 30,
            'expected_total_epsilon': 30,
        },
        {
            'testcase_name': 'advantage_over_basic1',
            'epsilon': 1,
            'delta': 0.001,
            'total_delta': 0.06,
            'num_queries': 30,
            'expected_total_epsilon': 22,
        },
        {
            'testcase_name': 'advantage_over_basic2',
            'epsilon': 1,
            'delta': 0.001,
            'total_delta': 0.1,
            'num_queries': 30,
            'expected_total_epsilon': 20,
        },
        {
            'testcase_name': 'total_delta_too_small',
            'epsilon': 1,
            'delta': 0.2,
            'total_delta': 0.1,
            'num_queries': 1,
            'expected_total_epsilon': None,
        },
        {
            'testcase_name': 'total_delta_too_small2',
            'epsilon': 1,
            'delta': 0.01,
            'total_delta': 0.26,
            'num_queries': 30,
            'expected_total_epsilon': None,
        })
    def test_advanced_composition(self, epsilon, delta, num_queries, total_delta,
                                  expected_total_epsilon):
        privacy_parameters = common.DifferentialPrivacyParameters(
            epsilon, delta)
        total_epsilon = accountant.advanced_composition(privacy_parameters,
                                                        num_queries, total_delta)
        if expected_total_epsilon is None:
            self.assertIsNone(total_epsilon)
        else:
            self.assertAlmostEqual(expected_total_epsilon, total_epsilon)

    @parameterized.named_parameters(
        {
            'testcase_name': 'basic_composition',
            'total_epsilon': 30,
            'total_delta': 0,
            'delta': 0,
            'num_queries': 30,
            'expected_epsilon': 1,
        },
        {
            'testcase_name': 'advantage_over_basic',
            'total_epsilon': 22,
            'total_delta': 0.06,
            'delta': 0.001,
            'num_queries': 30,
            'expected_epsilon': 1,
        },
        {
            'testcase_name': 'advantage_over_basic2',
            'total_epsilon': 5,
            'total_delta': 0.01,
            'delta': 0,
            'num_queries': 50,
            'expected_epsilon': 0.25,
        },
        {
            'testcase_name': 'total_delta_too_small',
            'total_epsilon': 1,
            'total_delta': 0.1,
            'delta': 0.2,
            'num_queries': 1,
            'expected_epsilon': None,
        },
        {
            'testcase_name': 'total_delta_too_small2',
            'total_epsilon': 30,
            'total_delta': 0.26,
            'delta': 0.01,
            'num_queries': 30,
            'expected_epsilon': None,
        })
    def test_get_smallest_epsilon_from_advanced_composition(
            self, total_epsilon, total_delta, num_queries, delta, expected_epsilon):
        total_privacy_parameters = common.DifferentialPrivacyParameters(
            total_epsilon, total_delta)
        epsilon = accountant.get_smallest_epsilon_from_advanced_composition(
            total_privacy_parameters, num_queries, delta)
        if expected_epsilon is None:
            self.assertIsNone(epsilon)
        else:
            self.assertAlmostEqual(expected_epsilon, epsilon, places=6)


if __name__ == '__main__':
    unittest.main()
