import copy
import unittest
from ....differential_privacy.solutions import rappor


class RapporParamsTest(unittest.TestCase):
    def setUp(self):
        self.typical_instance = rappor.Params(
            num_bloombits=16,
            num_hashes=2,
            num_cohorts=64,
            prob_p=0.40,
            prob_q=0.70,
            prob_f=0.30,
        )

    def test_getBloomBits(self):
        for cohort in range(0, 64):
            b = rappor.get_bloom_bits("foo", cohort, 2, 16)
            print(f"cohort={cohort}, bloom={b}")

    def test_getPrr(self):
        num_bits = 8
        for word in ("v1", "v2", "v3"):
            masks = rappor.get_prr_masks("secret", word, 0.5, num_bits)
            print(f"masks: {masks}")

    def test_toBigEndian(self):
        b = rappor.to_big_endian(1)
        # print(f"repr(b): {repr(b)}")
        self.assertEqual(4, len(b))

    def test_encoder(self):
        # Test encoder with deterministic random function.
        params = copy.copy(self.typical_instance)
        params.prob_f = 0.5
        params.prob_p = 0.5
        params.prob_q = 0.75

        # return these 3 probabilities in sequence.
        rand = MockRandom([0.0, 0.6, 0.0], params)
        e = rappor.Encoder(params, 0, "secret", rand)
        irr = e.encode("abc")
        self.assertEquals(64493, irr)  # given MockRandom, this is what we get


class MockRandom(object):
    """Returns one of three random values in a cyclic manner.

    Mock random function that involves *some* state, as needed for tests that
    call randomness several times. This makes it difficult to deal exclusively
    with stubs for testing purposes.
    """

    def __init__(self, cycle, params):
        self.p_gen = MockRandomCall(params.prob_p, cycle, params.num_bloombits)
        self.q_gen = MockRandomCall(params.prob_q, cycle, params.num_bloombits)


class MockRandomCall:
    def __init__(self, prob, cycle, num_bits):
        self.cycle = cycle
        self.n = len(self.cycle)
        self.prob = prob
        self.num_bits = num_bits

    def __call__(self):
        counter = 0
        r = 0
        for i in range(0, self.num_bits):
            rand_val = self.cycle[counter]
            counter += 1
            counter %= self.n  # wrap around
            r |= (rand_val < self.prob) << i
        return r


if __name__ == "__main__":
    unittest.main()
