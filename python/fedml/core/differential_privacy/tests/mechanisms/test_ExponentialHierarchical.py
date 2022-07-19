import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import ExponentialHierarchical
from fedml.core.differential_privacy.utils import global_seed


class TestExponentialHierarchical(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = ExponentialHierarchical

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMechanism
        self.assertTrue(issubclass(ExponentialHierarchical, DPMechanism))

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=-1, hierarchy=[])

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1 + 2j, hierarchy=[])

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon="Two", hierarchy=[])

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), hierarchy=[["A", "B"], ["C"]])

        for i in range(1000):
            self.assertEqual(mech.randomise("A"), "A")

    def test_non_string_hierarchy(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, hierarchy=[["A", "B"], ["C", 2]])

    def test_non_list_hierarchy(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, hierarchy=("A", "B", "C"))

    def test_uneven_hierarchy(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1, hierarchy=["A", ["B", "C"]])

    def test_build_utility_list(self):
        with self.assertRaises(TypeError):
            self.mech._build_utility_list([1, 2, 3])

    def test_non_string_input(self):
        mech = self.mech(epsilon=1, hierarchy=[["A", "B"], ["C", "2"]])
        with self.assertRaises(TypeError):
            mech.randomise(2)

    def test_outside_domain(self):
        mech = self.mech(epsilon=1, hierarchy=[["A", "B"], ["C"]])
        with self.assertRaises(ValueError):
            mech.randomise("D")

    def test_distrib_prob(self):
        epsilon = np.log(2)
        runs = 20000
        balanced_tree = False
        mech = self.mech(epsilon=epsilon, hierarchy=[["A", "B"], ["C"]])
        count = [0, 0, 0]

        for i in range(runs):
            val = mech.randomise("A")
            if val == "A":
                count[0] += 1
            elif val == "B":
                count[1] += 1
            elif val == "C":
                count[2] += 1

        # print(_mech.get_utility_list())
        # print(_mech._sensitivity)

        # print("A: %d, B: %d, C: %d" % (count[0], count[1], count[2]))
        self.assertAlmostEqual(count[0] / runs, np.exp(epsilon / (1 if balanced_tree else 2)) * count[2] / runs,
                               delta=0.1)
        self.assertAlmostEqual(count[0] / count[1], count[1] / count[2], delta=0.15)

    def test_neighbours_prob(self):
        epsilon = np.log(2)
        runs = 20000
        mech = self.mech(epsilon=epsilon, hierarchy=[["A", "B"], ["C"]])
        count = [0, 0, 0]

        for i in range(runs):
            val = mech.randomise("A")
            if val == "A":
                count[0] += 1

            val = mech.randomise("B")
            if val == "A":
                count[1] += 1

            val = mech.randomise("C")
            if val == "A":
                count[2] += 1

        # print("Output: A\nInput: A: %d, B: %d, C: %d" % (count[0], count[1], count[2]))
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs)
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[2] / runs)
        self.assertLessEqual(count[1] / runs, np.exp(epsilon) * count[2] / runs)

    def test_neighbours_flat_hierarchy_prob(self):
        epsilon = np.log(2)
        runs = 20000
        mech = self.mech(epsilon=epsilon, hierarchy=["A", "B", "C"])
        count = [0, 0, 0]

        for i in range(runs):
            val = mech.randomise("A")
            if val == "A":
                count[0] += 1

            val = mech.randomise("B")
            if val == "A":
                count[1] += 1

            val = mech.randomise("C")
            if val == "A":
                count[2] += 1

        # print("(Output: A) Input: A: %d, B: %d, C: %d" % (count[0], count[1], count[2]))
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.05)
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[2] / runs + 0.05)
        self.assertLessEqual(count[1] / runs, np.exp(epsilon) * count[2] / runs + 0.05)

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, hierarchy=[]))
        self.assertIn(".ExponentialHierarchical(", repr_)

    def test_bias(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, hierarchy=[]).bias, 0)

    def test_variance(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, hierarchy=[]).variance, 0)
