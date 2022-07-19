from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import TruncationAndFoldingMixin, DPMechanism


class TestTruncationAndFoldingMixin(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(TruncationAndFoldingMixin)

    def test_lone_instantiation(self):
        with self.assertRaises(TypeError):
            TruncationAndFoldingMixin(lower=0, upper=1)

    def test_dummy_instantiation(self):
        class TestClass(DPMechanism, TruncationAndFoldingMixin):
            def randomise(self, value):
                return 0

        mech = TestClass(epsilon=1, delta=0)
        self.assertEqual(mech.randomise(0), 0)
