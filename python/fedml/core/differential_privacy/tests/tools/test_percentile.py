from unittest import TestCase

import numpy as np
import pytest

from fedml.core.differential_privacy.tools.quantiles import percentile


class TestPercentile(TestCase):
    def test_not_none(self):
        mech = percentile
        self.assertIsNotNone(mech)

    def test_bad_percents(self):
        a = np.array([1, 2, 3])
        self.assertRaises(ValueError, percentile, a, -1)
        self.assertRaises(ValueError, percentile, a, 101)
        self.assertRaises(ValueError, percentile, a, [-1, 101])
        self.assertRaises(ValueError, percentile, a, [50] * 3 + [-1])

    def test_simple(self):
        a = np.random.random(1000)

        res = percentile(a, 50, epsilon=5, bounds=(0, 1))
        self.assertAlmostEqual(res, 0.5, delta=0.05)

    @pytest.mark.filterwarnings("ignore:Bounds have not been specified")
    def test_uniform_array(self):
        a = np.array([1] * 10)
        res = percentile(a, 20, epsilon=1)
        self.assertTrue(0 <= res <= 2)
