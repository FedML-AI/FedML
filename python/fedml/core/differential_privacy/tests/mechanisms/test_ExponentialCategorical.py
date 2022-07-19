import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import ExponentialCategorical
from fedml.core.differential_privacy.utils import global_seed


class TestExponential(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = ExponentialCategorical

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMechanism
        self.assertTrue(issubclass(ExponentialCategorical, DPMechanism))

    def test_inf_epsilon(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        mech = self.mech(epsilon=float("inf"), utility_list=utility_list)

        # print(_mech.randomise("A"))

        for i in range(1000):
            self.assertEqual(mech.randomise("A"), "A")

    def test_nonzero_delta(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        mech = self.mech(epsilon=1, utility_list=utility_list)
        mech.delta = 0.1

        with self.assertRaises(ValueError):
            mech.randomise("A")

    def test_non_string_hierarchy(self):
        utility_list = [
            ["A", "B", 1],
            ["A", 2, 2],
            ["B", 2, 2]
        ]
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, utility_list=utility_list)

    def test_missing_utilities(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2]
        ]
        with self.assertRaises(ValueError):
            self.mech(epsilon=1, utility_list=utility_list)

    def test_wrong_utilities(self):
        utility_list = (
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        )
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, utility_list=utility_list)

        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", "2"]
        ]
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, utility_list=utility_list)

        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", -2]
        ]

        with self.assertRaises(ValueError):
            self.mech(epsilon=1, utility_list=utility_list)

    def test_non_string_input(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        mech = self.mech(epsilon=1, utility_list=utility_list)
        with self.assertRaises(TypeError):
            mech.randomise(2)

    def test_outside_domain(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        mech = self.mech(epsilon=1, utility_list=utility_list)
        with self.assertRaises(ValueError):
            mech.randomise("D")

    def test_get_utility_list(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["C", "B", 2]
        ]
        mech = self.mech(epsilon=1, utility_list=utility_list)

        _utility_list = mech.utility_list
        self.assertEqual(len(_utility_list), len(utility_list))

    def test_self_in_utility(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2],
            ["A", "A", 5]
        ]
        mech = self.mech(epsilon=1, utility_list=utility_list)

        _utility_list = mech.utility_list
        self.assertEqual(len(_utility_list) + 1, len(utility_list))

        self.assertEqual(mech._get_utility("A", "A"), 0)

    def test_distrib_prob(self):
        epsilon = np.log(2)
        runs = 20000
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        mech = self.mech(epsilon=epsilon, utility_list=utility_list)
        count = [0, 0, 0]

        for i in range(runs):
            val = mech.randomise("A")
            if val == "A":
                count[0] += 1
            elif val == "B":
                count[1] += 1
            elif val == "C":
                count[2] += 1

        # print("A: %d, B: %d, C: %d" % (count[0], count[1], count[2]))
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[2] / runs + 0.05)
        self.assertAlmostEqual(count[0] / count[1], count[1] / count[2], delta=0.15)

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, utility_list=[]))
        self.assertIn(".ExponentialCategorical(", repr_)

    def test_bias(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, utility_list=[]).bias, 0)

    def test_variance(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, utility_list=[]).variance, 0)
