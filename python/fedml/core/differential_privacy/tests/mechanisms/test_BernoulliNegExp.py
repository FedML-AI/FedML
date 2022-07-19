import secrets

import numpy as np
from unittest import TestCase

from fedml.core.differential_privacy.mechanisms.base import bernoulli_neg_exp


class TestBernoulliNegExp(TestCase):
    def setup_method(self, method):
        self.rng = secrets.SystemRandom()

    def teardown_method(self, method):
        del self.rng

    def test_bad_gamma(self):
        self.assertRaises(ValueError, bernoulli_neg_exp, -1)
        self.assertRaises(TypeError, bernoulli_neg_exp, 1j)

    def test_infinite_gamma(self):
        for i in range(1000):
            self.assertEqual(0, bernoulli_neg_exp(float("inf"), self.rng))

    def test_zero_gamma(self):
        for i in range(1000):
            self.assertEqual(1, bernoulli_neg_exp(0, self.rng))

    def test_no_rng(self):
        for i in range(100):
            self.assertIn(bernoulli_neg_exp(1), [0, 1])

    def test_bad_rng(self):
        self.assertRaises(AttributeError, bernoulli_neg_exp, 1, "rng")

    def test_output_domain(self):
        for i in range(1000):
            self.assertIn(bernoulli_neg_exp(1, self.rng), [0, 1])

    def test_bernoulli_exp_prob(self):
        runs = 5000

        self.assertAlmostEqual(sum(bernoulli_neg_exp(-np.log(0.5), self.rng) for _ in range(runs)) / runs,
                               0.5, delta=0.025)
        self.assertAlmostEqual(sum(bernoulli_neg_exp(-np.log(0.1), self.rng) for _ in range(runs)) / runs,
                               0.1, delta=0.025)
