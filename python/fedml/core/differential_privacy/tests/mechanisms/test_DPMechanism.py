from unittest import TestCase

from fedml.core.differential_privacy.mechanisms import DPMachine, DPMechanism


class TestDPMechanism(TestCase):
    def setup_method(self, method):
        class TestMech(DPMechanism):
            def randomise(self, value):
                return value

        self.mech = TestMech

    def teardown_method(self, method):
        del self.mech

    def test_not_none(self):
        self.assertIsNotNone(DPMechanism)

    def test_parent_class(self):
        self.assertTrue(issubclass(DPMechanism, DPMachine))

    def test_instantiation(self):
        with self.assertRaises(TypeError):
            DPMechanism(epsilon=1, delta=0)

    def test_copy(self):
        self.assertIsInstance(self.mech(epsilon=1, delta=0).copy(), DPMechanism)

    def test_set_epsilon_delta(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=0, delta=-1)

        with self.assertRaises(ValueError):
            self.mech(epsilon=0, delta=2)

        with self.assertRaises(ValueError):
            self.mech(epsilon=0, delta=0)

        mech1 = self.mech(epsilon=1, delta=0).copy()
        self.assertIsNotNone(mech1)

    def test_bias(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, delta=0).bias, 1)

    def test_variance(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, delta=0).variance, 1)

    def test_mse(self):
        self.assertRaises(NotImplementedError, self.mech(epsilon=1, delta=0).mse, 1)

        class TestMSE(self.mech):
            def bias(self, value):
                return -1

            def variance(self, value):
                return 1

        mse_mech = TestMSE(epsilon=1, delta=0)

        self.assertEqual(mse_mech.mse(1), 2)
