"""Tests for test_util."""

import unittest
from absl.testing import parameterized

from dp_accounting.pld import test_util


class TestUtilTest(parameterized.TestCase):

  @parameterized.parameters(
      # Dictionary contained
      ({
          1: 0.1,
          2: 0.3
      }, {
          1: 0.1,
          2: 0.3,
          4: 0.2
      }, True),
      ({
          1: 1e-10,
          2: 0.3
      }, {
          2: 0.3,
          4: 0.2
      }, True),
      ({
          1.0: 0.1 + 1e-10,
          2.0: 0.3
      }, {
          1.0 + 1e-10: 0.1,
          2.0: 0.3 + 1e-10,
          4.0: 0.2
      }, True),
      # Dictionary not contained
      ({
          1: 0.1,
          2: 0.3
      }, {
          2: 0.3,
          4: 0.2
      }, False))
  def test_assert_dictionary_contained(self, dict1, dict2, expected_result):
    if expected_result:
      test_util.assert_dictionary_contained(self, dict1, dict2)
    else:
      with self.assertRaises(AssertionError):
        test_util.assert_dictionary_contained(self, dict1, dict2)

  @parameterized.parameters(
      # Dictionary almost equal
      ({
          1: 0.1,
          2: 0.3,
      }, {
          1: 0.1,
          2: 0.3
      }, True),
      ({
          1: 1e-10,
          2: 0.3,
          4: 0.2,
      }, {
          2: 0.3,
          4: 0.2
      }, True),
      ({
          1.0: 0.1 + 1e-10,
          2.0: 0.3,
          4.0 + 1e-10: 0.2
      }, {
          1.0 + 1e-10: 0.1,
          2.0: 0.3 + 1e-10,
          4.0: 0.2 - 1e-10
      }, True),
      # Dictionary not almost equal
      ({
          1: 0.1,
          2: 0.3,
      }, {
          2: 0.3,
          4: 0.2
      }, False))
  def test_dictionary_almost_equal(self, dict1, dict2, expected_result):
    if expected_result:
      test_util.assert_dictionary_almost_equal(self, dict1, dict2)
    else:
      with self.assertRaises(AssertionError):
        test_util.assert_dictionary_almost_equal(self, dict1, dict2)

  @parameterized.parameters(
      (2, 1, True), (2, 2+1e-10, True), (2+1e-10, 2, True),
      (1, 2, False))
  def test_assert_almost_greater_equal(self, a, b, expected_result):
    if expected_result:
      test_util.assert_almost_greater_equal(self, a, b)
    else:
      with self.assertRaises(AssertionError):
        test_util.assert_almost_greater_equal(self, a, b)


if __name__ == '__main__':
  unittest.main()
