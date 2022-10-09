"""Helper functions for testing.
"""

from typing import Optional, Mapping, Union
import unittest  # pylint:disable=unused-import

import numpy as np


def assert_dictionary_contained(testcase: 'unittest.TestCase',
                                dict1: Mapping[Union[int, float], float],
                                dict2: Mapping[Union[int, float], float]):
  """Check whether first dictionary is contained in the second.

  Keys of type float are checked for almost equality. Values are always checked
  for almost equality.

  Keys corresponding to values close to 0 are ignored in this test.

  Args:
    testcase: unittestTestCase object to assert containment of dictionary.
    dict1: first dictionary
    dict2: second dictionary
  """
  for i in dict1.keys():
    if not np.isclose(dict1[i], 0):
      found = False
      for j in dict2.keys():
        if np.isclose(i, j) and np.isclose(dict1[i], dict2[j]):
          found = True
          break
      testcase.assertTrue(found, msg=f'Key {i} in {dict1} not found in {dict2}')


def assert_dictionary_almost_equal(testcase: 'unittest.TestCase',
                                   dictionary1: Mapping[Union[int, float],
                                                        float],
                                   dictionary2: Mapping[Union[int, float],
                                                        float]):
  """Check two dictionaries have almost equal values.

  Keys of type float are checked for almost equality. Values are always checked
  for almost equality.

  Keys corresponding to values close to 0 are ignored in this test.

  Args:
    testcase: unittestTestCase object to assert containment of dictionary.
    dictionary1: first dictionary
    dictionary2: second dictionary
  """
  assert_dictionary_contained(testcase, dictionary1, dictionary2)
  assert_dictionary_contained(testcase, dictionary2, dictionary1)


def assert_almost_greater_equal(testcase: 'unittest.TestCase',
                                a: float, b: float, msg: Optional[str] = None):
  """Asserts that first value is greater or almost equal to second value."""
  msg = f'{a} is less than {b}' if msg is None else msg
  testcase.assertTrue(a >= b or np.isclose(a, b), msg=msg)
