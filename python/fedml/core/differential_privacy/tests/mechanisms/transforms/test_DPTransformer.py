from numbers import Real
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import ExponentialHierarchical, Laplace
from fedml.core.differential_privacy.mechanisms.transforms import DPTransformer


class TestDPTransformer(TestCase):
    def test_not_none(self):
        mech = DPTransformer(ExponentialHierarchical(epsilon=1, hierarchy=["A", "B"]))
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from fedml.core.differential_privacy.mechanisms import DPMachine
        self.assertTrue(issubclass(DPTransformer, DPMachine))

    def test_no_parent(self):
        with self.assertRaises(TypeError):
            DPTransformer()

    def test_bad_parent(self):
        with self.assertRaises(TypeError):
            DPTransformer(int)

    def test_nested(self):
        mech = DPTransformer(DPTransformer(DPTransformer(ExponentialHierarchical(epsilon=1, hierarchy=["A", "B"]))))
        self.assertIsNotNone(mech)

    def test_laplace(self):
        mech = DPTransformer(Laplace(epsilon=1, sensitivity=1))
        self.assertIsInstance(mech.randomise(1), Real)
