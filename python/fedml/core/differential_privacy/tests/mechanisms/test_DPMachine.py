import abc
from unittest import TestCase

from numpy.random import random

from fedml.core.differential_privacy.mechanisms import DPMachine


class TestDPMachine(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(DPMachine)

    def test_class(self):
        self.assertTrue(issubclass(DPMachine, abc.ABC))

    def test_instantiation(self):
        with self.assertRaises(TypeError):
            DPMachine()

    def test_base(self):
        class BaseDPMachine(DPMachine):
            def _check_epsilon(self, epsilon):
                return self

            def _check_epsilon_delta(self, epsilon, delta):
                return self

            def randomise(self, value):
                return random()

        mech = BaseDPMachine()
        val = mech.randomise(1)

        self.assertIsInstance(val, float)
