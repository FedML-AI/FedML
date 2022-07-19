import math
import sys

import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import Snapping
from fedml.core.differential_privacy.utils import global_seed


class TestSnapping(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Snapping

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMechanism
        self.assertTrue(issubclass(Snapping, DPMechanism))

    def test_neg_sensitivity(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1.0, sensitivity=-1, lower=0, upper=1000)

    def test_str_sensitivity(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1.0, sensitivity="1", lower=0, upper=1000)

    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1.0, sensitivity=0, lower=0, upper=1000)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=-1, sensitivity=1, lower=0, upper=1000)

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), sensitivity=1, lower=0, upper=1000)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1 + 2j, sensitivity=1, lower=0, upper=1000)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon="Two", sensitivity=1, lower=0, upper=1000)

    def test_epsilon(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        self.assertIsNotNone(mech.randomise(1))

    def test_neg_effective_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=np.nextafter((2*np.finfo(float).epsneg), 0), sensitivity=1, lower=0, upper=1000)

    def test_zero_effective_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=2 * np.finfo(float).epsneg, sensitivity=1, lower=0, upper=1000)

    def test_immediately_above_zero_effective_epsilon(self):
        mech = self.mech(epsilon=np.nextafter(2 * np.finfo(float).epsneg, math.inf), sensitivity=1, lower=0, upper=1000)
        self.assertIsNotNone(mech.randomise(1))

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000))
        self.assertIn(".Snapping(", repr_)

    def test_bias(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        with self.assertRaises(NotImplementedError):
            mech.bias(1)

    def test_variance(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        with self.assertRaises(NotImplementedError):
            mech.variance(1)

    def test_scale_bound_symmetric_sensitivity_1(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=-10, upper=10)
        self.assertAlmostEqual(mech._scale_bound(), 10)

    def test_scale_bound_nonsymmetric_sensitivity_1(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=-10, upper=30)
        self.assertAlmostEqual(mech._scale_bound(), 20)

    def test_scale_bound_nonsymmetric_sensitivity_0(self):
        mech = self.mech(epsilon=1.0, sensitivity=0, lower=-10, upper=30)
        self.assertAlmostEqual(mech._scale_bound(), 20)

    def test_scale_bound_nonsymmetric_sensitivity_subunitary(self):
        mech = self.mech(epsilon=1.0, sensitivity=0.1, lower=-10, upper=30)
        self.assertAlmostEqual(mech._scale_bound(), 200)

    def test_scale_bound_nonsymmetric_sensitivity_supraunitary(self):
        mech = self.mech(epsilon=1.0, sensitivity=2, lower=-10, upper=30)
        self.assertAlmostEqual(mech._scale_bound(), 10)

    def test_scale_and_offset_value_symmetric_sensitivity_1(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=-10, upper=10)
        self.assertAlmostEqual(mech._scale_and_offset_value(10), 10)

    def test_scale_and_offset_value_nonsymmetric_sensitivity_1(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=-10, upper=30)
        self.assertAlmostEqual(mech._scale_and_offset_value(10), 0)

    def test_scale_and_offset_value_nonsymmetric_sensitivity_1_lower_bound(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=-10, upper=30)
        self.assertAlmostEqual(mech._scale_and_offset_value(-10), -20)

    def test_scale_and_offset_value_nonsymmetric_sensitivity_1_upper_bound(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=10, upper=30)
        self.assertAlmostEqual(mech._scale_and_offset_value(30), 10)

    def test_scale_and_offset_value_symmetric_sensitivity_subunitary(self):
        mech = self.mech(epsilon=1.0, sensitivity=0.1, lower=-10, upper=10)
        self.assertAlmostEqual(mech._scale_and_offset_value(10), 100)

    def test_scale_and_offset_value_symmetric_sensitivity_supraunitary(self):
        mech = self.mech(epsilon=1.0, sensitivity=2, lower=-10, upper=10)
        self.assertAlmostEqual(mech._scale_and_offset_value(10), 5)

    def test_reverse_scale_and_offset_value_symmetric_sensitivity_1(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=-10, upper=10)
        self.assertAlmostEqual(mech._reverse_scale_and_offset_value(10), 10)

    def test_reverse_scale_and_offset_value_nonsymmetric_sensitivity_1(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=-10, upper=30)
        self.assertAlmostEqual(mech._reverse_scale_and_offset_value(0), 10)

    def test_reverse_scale_and_offset_value_nonsymmetric_sensitivity_1_lower_bound(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=-10, upper=30)
        self.assertAlmostEqual(mech._reverse_scale_and_offset_value(-20), -10)

    def test_reverse_scale_and_offset_value_nonsymmetric_sensitivity_1_upper_bound(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=10, upper=30)
        self.assertAlmostEqual(mech._reverse_scale_and_offset_value(10), 30)

    def test_reverse_scale_and_offset_value_symmetric_sensitivity_subunitary(self):
        mech = self.mech(epsilon=1.0, sensitivity=0.1, lower=-10, upper=10)
        self.assertAlmostEqual(mech._reverse_scale_and_offset_value(100), 10)

    def test_reverse_scale_and_offset_value_symmetric_sensitivity_supraunitary(self):
        mech = self.mech(epsilon=1.0, sensitivity=2, lower=-10, upper=10)
        self.assertAlmostEqual(mech._reverse_scale_and_offset_value(5), 10)

    def test_get_nearest_power_of_2_exact(self):
        self.assertEqual(Snapping._get_nearest_power_of_2(2 ** 2), 4)
        self.assertEqual(Snapping._get_nearest_power_of_2(2 ** 0), 1)
        self.assertEqual(Snapping._get_nearest_power_of_2(2 ** -2), 0.25)
        self.assertEqual(Snapping._get_nearest_power_of_2(0), 0)

    def test_get_nearest_power_of_2(self):
        self.assertEqual(Snapping._get_nearest_power_of_2(np.nextafter(2.0, math.inf)), 4)
        self.assertEqual(Snapping._get_nearest_power_of_2(np.nextafter(2.0, -math.inf)), 2)
        self.assertEqual(Snapping._get_nearest_power_of_2(np.nextafter(sys.float_info.min, -math.inf)),
                               sys.float_info.min)

    def test_round_to_nearest_power_of_2_exact(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        self.assertAlmostEqual(mech._round_to_nearest_power_of_2(2.0, 2.0), 2)

    def test_round_to_nearest_power_of_2_below_exact(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        self.assertAlmostEqual(mech._round_to_nearest_power_of_2(np.nextafter(2.0, -math.inf), 2.0), 2)

    def test_round_to_nearest_power_of_2_above_exact(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        self.assertAlmostEqual(mech._round_to_nearest_power_of_2(np.nextafter(2.0, math.inf), 2.0), 2.0)

    def test_round_to_nearest_power_of_2_middle(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        self.assertAlmostEqual(mech._round_to_nearest_power_of_2(3.0, 2.0), 4)

    def test_round_to_nearest_power_of_2_below_middle(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        self.assertAlmostEqual(mech._round_to_nearest_power_of_2(np.nextafter(3.0, -math.inf), 2.0), 2)

    def test_round_to_nearest_power_of_2_above_middle(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        self.assertAlmostEqual(mech._round_to_nearest_power_of_2(np.nextafter(3.0, math.inf), 2.0), 4.0)

    def test_non_numeric(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_zero_median_prob(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1000)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)

    def test_effective_epsilon(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1)

        self.assertLess(mech.effective_epsilon(), 1.0)

    def test_within_bounds(self):
        mech = self.mech(epsilon=1.0, sensitivity=1, lower=0, upper=1)
        vals = []

        for i in range(1000):
            vals.append(mech.randomise(0.5))

        vals = np.array(vals)

        self.assertTrue(np.all(vals >= 0))
        self.assertTrue(np.all(vals <= 1))

    def test_neighbours_prob(self):
        epsilon = 1
        runs = 10000
        mech = self.mech(epsilon=epsilon, sensitivity=1, lower=0, upper=1000)
        count = [0, 0]

        for i in range(runs):
            val0 = mech.randomise(0)
            if val0 <= 0:
                count[0] += 1

            val1 = mech.randomise(1)
            if val1 <= 0:
                count[1] += 1

        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.1)
