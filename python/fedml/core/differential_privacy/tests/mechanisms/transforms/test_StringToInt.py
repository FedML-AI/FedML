import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import Geometric
from fedml.core.differential_privacy.mechanisms.transforms import StringToInt
from fedml.core.differential_privacy.utils import global_seed

global_seed(3141592653)


class TestStringToInt(TestCase):
    def test_not_none(self):
        mech = StringToInt(Geometric(epsilon=1))
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMachine
        from fedml.core.differential_privacy.mechanisms.transforms import DPTransformer

        self.assertTrue(issubclass(StringToInt, DPMachine))
        self.assertTrue(issubclass(StringToInt, DPTransformer))

    def test_no_parent(self):
        with self.assertRaises(TypeError):
            StringToInt()

    def test_randomise(self):
        mech = StringToInt(Geometric(epsilon=1))
        self.assertIsInstance(mech.randomise("1"), str)

    def test_distrib(self):
        epsilon = 1.0
        runs = 10000
        mech = StringToInt(Geometric(epsilon=epsilon))
        count = [0, 0]

        for _ in range(runs):
            if mech.randomise("0") == "0":
                count[0] += 1

            if mech.randomise("1") == "0":
                count[1] += 1

        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0] / runs, count[1] * np.exp(epsilon) / runs + 0.05)
