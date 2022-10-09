"""Tests for privacy_loss_mechanism."""

import math
import unittest
from absl.testing import parameterized
from scipy import stats

from dp_accounting.pld import common
from dp_accounting.pld import privacy_loss_mechanism
from dp_accounting.pld import test_util


ADD = privacy_loss_mechanism.AdjacencyType.ADD
REM = privacy_loss_mechanism.AdjacencyType.REMOVE


class LaplacePrivacyLossTest(parameterized.TestCase):

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, -0.1, 1.0), (1.0, 1.0, 1.0, ADD, 2.0, -1.0),
      (1.0, 1.0, 1.0, ADD, 0.3, 0.4), (4.0, 4.0, 1.0, ADD, -0.4, 1.0),
      (5.0, 5.0, 1.0, ADD, 7.0, -1.0), (7.0, 7.0, 1.0, ADD, 2.1, 0.4),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 1.1, -0.86483972516319),
      (2.0, 1.0, 0.2, ADD, -0.2, 0.0819629071393439),
      (1.0, 1.0, 0.5, ADD, 0.5, 0.0),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, -1.1, 1.0), (1.0, 1.0, 1.0, REM, 1.0, -1.0),
      (1.0, 1.0, 1.0, REM, -0.7, 0.4), (4.0, 4.0, 1.0, REM, -4.4, 1.0),
      (5.0, 5.0, 1.0, REM, 2.0, -1.0), (7.0, 7.0, 1.0, REM, -4.9, 0.4),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, -1.1, 0.86483972516319),
      (2.0, 1.0, 0.2, REM, 0.2, -0.0819629071393439),
      (1.0, 1.0, 0.5, REM, -0.5, 0.0))
  def test_laplace_privacy_loss(self, parameter, sensitivity, sampling_prob,
                                adjacency_type, x, expected_privacy_loss):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 1.0, 0.0), (1.0, 1.0, 1.0, ADD, -1.0, math.inf),
      (1.0, 1.0, 1.0, ADD, 0.4, 0.3), (4.0, 4.0, 1.0, ADD, 1.0, 0.0),
      (5.0, 5.0, 1.0, ADD, -1.0, math.inf), (7.0, 7.0, 1.0, ADD, 0.4, 2.1),
      (1.0, 1.0, 1.0, ADD, 2.0, -math.inf), (3, 1, 1, ADD, 3.1, -math.inf),
      (4.0, 4.0, 1.0, ADD, 1.1, -math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, -0.8649, math.inf),
      (1.0, 1.0, 0.7, ADD, 1.0, -math.inf),
      (2.0, 1.0, 0.2, ADD, 0.0819629071393439, 0),
      (1.0, 1.0, 0.5, ADD, 0.0, 0.5),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, 1.0, -1.0), (1.0, 1.0, 1.0, REM, -1.0, math.inf),
      (1.0, 1.0, 1.0, REM, 0.4, -0.7), (4.0, 4.0, 1.0, REM, 1.0, -4.0),
      (5.0, 5.0, 1.0, REM, -1.0, math.inf), (7.0, 7.0, 1.0, REM, 0.4, -4.9),
      (1.0, 1.0, 1.0, REM, 2.0, -math.inf),
      (3.0, 1.0, 1.0, REM, 3.1, -math.inf),
      (4.0, 4.0, 1.0, REM, 1.1, -math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, 0.86483972516319, -1.0),
      (2.0, 1.0, 0.2, REM, -0.082, math.inf),
      (1.0, 1.0, 0.7, REM, 1.0, -math.inf),
      (1.0, 1.0, 0.5, REM, 0.0, -0.5))
  def test_laplace_inverse_privacy_loss(self, parameter, sensitivity,
                                        sampling_prob, adjacency_type,
                                        privacy_loss, expected_x):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 0.0, 1.0, {1: 0.5, -1: 0.18393972}),
      (3.0, 3.0, 1.0, ADD, 0.0, 3.0, {1: 0.5, -1: 0.18393972}),
      (1.0, 2.0, 1.0, ADD, 0.0, 2.0, {2: 0.5, -2: 0.06766764}),
      (4.0, 8.0, 1.0, ADD, 0.0, 8.0, {2: 0.5, -2: 0.06766764}),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 0.0, 1.0, {
          0.7046054708796524: 0.5,
          -0.864839725163191: 0.18393972
      }),
      (3.0, 3.0, 0.6, ADD, 0.0, 3.0, {
          0.4768628363884146: 0.5,
          -0.7085130668623151: 0.18393972
      }),
      (1.0, 2.0, 0.7, ADD, 0.0, 2.0, {
          0.929541389699331: 0.5,
          -1.699706179357965: 0.06766764
      }),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, -1.0, 0.0, {1: 0.5, -1: 0.18393972}),
      (3.0, 3.0, 1.0, REM, -3.0, 0.0, {1: 0.5, -1: 0.18393972}),
      (1.0, 2.0, 1.0, REM, -2.0, 0.0, {2: 0.5, -2: 0.06766764}),
      (4.0, 8.0, 1.0, REM, -8.0, 0.0, {2: 0.5, -2: 0.06766764}),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, -1.0, 0.0, {
          0.864839725163191: 0.4367879441171443,
          -0.7046054708796524: 0.2471517764685769
      }),
      (3.0, 3.0, 0.6, REM, -3.0, 0.0, {
          0.7085130668623151: 0.3735758882342885,
          -0.4768628363884146: 0.3103638323514328
      }),
      (1.0, 2.0, 0.7, REM, -2.0, 0.0, {
          1.699706179357965: 0.3703002924854919,
          -0.929541389699331: 0.1973673491328145
      }))
  def test_laplace_privacy_loss_tail(self, parameter, sensitivity,
                                     sampling_prob, adjacency_type,
                                     expected_lower_x_truncation,
                                     expected_upper_x_truncation,
                                     expected_tail_probability_mass_function):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.assert_dictionary_almost_equal(
        self, expected_tail_probability_mass_function,
        tail_pld.tail_probability_mass_function)

  @parameterized.parameters((-3.0, 1.0, 1.0, ADD), (0.0, 1.0, 1.0, ADD),
                            (1.0, 0.0, 1.0, REM), (2.0, -1.0, 1.0, REM),
                            (2.0, 1.0, 0.0, ADD), (1.0, 1.0, 1.2, REM),
                            (2.0, 1.0, -0.1, REM))
  def test_laplace_value_errors(self,
                                parameter,
                                sensitivity,
                                sampling_prob=1.0,
                                adjacency_type=ADD):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.LaplacePrivacyLoss(
          parameter,
          sensitivity=sensitivity,
          sampling_prob=sampling_prob,
          adjacency_type=adjacency_type)

  @parameterized.parameters((1.0, 1.0, 1.0, 1.1), (1.0, 1.0, 1.0, -0.1),
                            (1.0, 0.0, 1.0, 0.1), (1.0, -0.2, 1.0, 0.1),
                            (1.0, 1.1, 1.0, 0.2))
  def test_laplace_from_privacy_parameters_value_errors(
      self, sensitivity, sampling_prob, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.GaussianPrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity, sampling_prob=sampling_prob)

  @parameterized.parameters((1.0, 1.0, ADD, 1.0, 0.0, 1.0),
                            (1.0, 1.0, ADD, 1.0, 0.1, 1.0),
                            (2.0, 1.0, REM, 1.0, 0.01, 2.0),
                            (1.0, 1.0, REM, 3.0, 0.01, 0.33333333),
                            (1.0, 0.8, ADD, 1.0, 0.0, 0.8720521537764049),
                            (1.0, 0.5, REM, 1.0, 0.1, 0.671194938966816),
                            (2.0, 0.9, ADD, 1.0, 0.01, 1.8728716669259162),
                            (1.0, 0.7, REM, 3.0, 0.01, 0.2992554981396725))
  def test_laplace_from_privacy_parameters(self, sensitivity, sampling_prob,
                                           adjacency_type,
                                           epsilon, delta,
                                           expected_parameter):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss.from_privacy_guarantee(
        common.DifferentialPrivacyParameters(epsilon, delta),
        sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_parameter, pl.parameter)
    self.assertEqual(adjacency_type, pl.adjacency_type)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 1.0, 0.0), (3.0, 3.0, 1.0, ADD, 1.0, 0.0),
      (2.0, 4.0, 1.0, ADD, 2.0, 0.0),
      (2.0, 4.0, 1.0, ADD, 0.5, 0.52763345),
      (1.0, 1.0, 1.0, ADD, 0.0, 0.39346934),
      (2.0, 2.0, 1.0, ADD, 0.0, 0.39346934),
      (1.0, 1.0, 1.0, ADD, -2.0, 0.86466472),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 1.0, 0.0),
      (2.0, 4.0, 0.8, ADD, 0.5, 0.3243606497234246),
      (1.0, 1.0, 0.6, ADD, 0.2, 0.1401134521354217),
      (2.0, 2.0, 0.3, ADD, 0.0, 0.1180408020862099),
      (5.0, 5.0, 0.2, ADD, 2.0, 0.0),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, 1.0, 0.0), (3.0, 3.0, 1.0, REM, 1.0, 0.0),
      (2.0, 4.0, 1.0, REM, 2.0, 0.0), (2.0, 4.0, 1.0, REM, 0.5, 0.52763345),
      (1.0, 1.0, 1.0, REM, 0.0, 0.39346934),
      (2.0, 2.0, 1.0, REM, 0.0, 0.39346934),
      (1.0, 1.0, 1.0, REM, -2.0, 0.86466472),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE)
      (1.0, 1.0, 0.8, REM, 1.0, 0.0),
      (2.0, 4.0, 0.8, REM, 0.5, 0.4039564635032081),
      (1.0, 1.0, 0.6, REM, 0.2, 0.1741992102060086),
      (2.0, 2.0, 0.3, REM, 0.0, 0.1180408020862099),
      (5.0, 5.0, 0.2, REM, -0.25, 0.2211992169285951))
  def test_laplace_get_delta_for_epsilon(self, parameter, sensitivity,
                                         sampling_prob, adjacency_type, epsilon,
                                         expected_delta):
    pl = privacy_loss_mechanism.LaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_delta, pl.get_delta_for_epsilon(epsilon))


class GaussianPrivacyLossTest(parameterized.TestCase):

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 5.0, -4.5), (1.0, 1.0, 1.0, ADD, -3.0, 3.5),
      (1.0, 2.0, 1.0, ADD, 3.0, -4.0),
      (4.0, 4.0, 1.0, ADD, 20.0, -4.5), (5.0, 5.0, 1.0, ADD, -15.0, 3.5),
      (7.0, 14.0, 1.0, ADD, 21.0, -4.0),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 0.5, 0.0),
      (1.0, 1.0, 0.5, ADD, -4, 0.6820994357113515),
      (1.0, 2.0, 0.7, ADD, 0, 0.929541389699331),
      (4.0, 4.0, 0.3, ADD, -16, 0.3519252431310541),
      (5.0, 5.0, 0.45, ADD, 20, -2.737735427805667),
      (7.0, 14.0, 0.9, ADD, -7, 2.150000710600199),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, 4.0, -4.5), (1.0, 1.0, 1.0, REM, -4.0, 3.5),
      (1.0, 2.0, 1.0, REM, 1.0, -4.0), (4.0, 4.0, 1.0, REM, 16.0, -4.5),
      (5.0, 5.0, 1.0, REM, -20.0, 3.5), (7.0, 14.0, 1.0, REM, 7.0, -4.0),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, -0.5, 0.0),
      (1.0, 1.0, 0.5, REM, 4.0, -0.6820994357113515),
      (1.0, 2.0, 0.7, REM, 0.0, -0.929541389699331),
      (4.0, 4.0, 0.3, REM, 16.0, -0.3519252431310541),
      (5.0, 5.0, 0.45, REM, -20.0, 2.737735427805667),
      (7.0, 14.0, 0.9, REM, 7.0, -2.150000710600199))
  def test_gaussian_privacy_loss(self, standard_deviation, sensitivity,
                                 sampling_prob, adjacency_type, x,
                                 expected_privacy_loss):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, -4.5, 5.0), (1.0, 1.0, 1.0, ADD, 3.5, -3.0),
      (1.0, 2.0, 1.0, ADD, -4.0, 3.0),
      (4.0, 4.0, 1.0, ADD, -4.5, 20.0), (5.0, 5.0, 1.0, ADD, 3.5, -15.0),
      (7.0, 14.0, 1.0, ADD, -4.0, 21.0),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 0.0, 0.5),
      (1.0, 1.0, 0.5, ADD, 0.6820994357113515, -4.0),
      (1.0, 2.0, 0.7, ADD, 0.929541389699331, 0.0),
      (4.0, 4.0, 0.3, ADD, 0.3519252431310541, -16.0),
      (5.0, 5.0, 0.45, ADD, -2.737735427805667, 20.0),
      (7.0, 14.0, 0.9, ADD, 2.150000710600199, -7.0),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, -4.5, 4.0), (1.0, 1.0, 1.0, REM, 3.5, -4.0),
      (1.0, 2.0, 1.0, REM, -4.0, 1.0), (4.0, 4.0, 1.0, REM, -4.5, 16.0),
      (5.0, 5.0, 1.0, REM, 3.5, -20.0), (7.0, 14.0, 1.0, REM, -4.0, 7.0),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, 0.0, -0.5),
      (1.0, 1.0, 0.5, REM, -0.6820994357113515, 4.0),
      (1.0, 2.0, 0.7, REM, -0.929541389699331, 0.0),
      (4.0, 4.0, 0.3, REM, -0.3519252431310541, 16.0),
      (5.0, 5.0, 0.45, REM, 2.737735427805667, -20.0),
      (7.0, 14.0, 0.9, REM, -2.150000710600199, 7.0))
  def test_gaussian_inverse_privacy_loss(self, standard_deviation, sensitivity,
                                         sampling_prob, adjacency_type,
                                         privacy_loss, expected_x):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, -1.0, 2.0, True, {
          math.inf: 0.15865525,
          -1.5: 0.02275013
      }),
      (3.0, 3.0, 1.0, ADD, -3.0, 6.0, True, {
          math.inf: 0.15865525,
          -1.5: 0.02275013
      }),
      (1.0, 2.0, 1.0, ADD, -1.0, 3.0, True, {
          math.inf: 0.15865525,
          -4.0: 0.00134989
      }),
      (4.0, 8.0, 1.0, ADD, -4.0, 12.0, True, {
          math.inf: 0.15865525,
          -4.0: 0.00134989
      }),
      (1.0, 1.0, 1.0, ADD, -1.0, 2.0, False, {
          1.5: 0.15865525,
      }),
      (3.0, 3.0, 1.0, ADD, -3.0, 6.0, False, {
          1.5: 0.15865525,
      }),
      (1.0, 2.0, 1.0, ADD, -1.0, 3.0, False, {
          4.0: 0.15865525,
      }),
      (4.0, 8.0, 1.0, ADD, -4.0, 12.0, False, {
          4.0: 0.15865525,
      }),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, -1.0, 2.0, True, {
          math.inf: 0.15865525,
          -1.331139: 0.02275013
      }),
      (3.0, 3.0, 0.8, ADD, -3.0, 6.0, True, {
          math.inf: 0.15865525,
          -1.331139: 0.02275013
      }),
      (1.0, 2.0, 0.5, ADD, -1.0, 3.0, True, {
          math.inf: 0.15865525,
          -3.325003: 0.00134990
      }),
      (4.0, 8.0, 0.6, ADD, -4.0, 12.0, True, {
          math.inf: 0.15865525,
          -3.501311: 0.00134990
      }),
      (1.0, 1.0, 0.9, ADD, -1.0, 2.0, False, {
          1.20125: 0.15865525,
      }),
      (3.0, 3.0, 0.7, ADD, -3.0, 6.0, False, {
          0.784843: 0.15865525,
      }),
      (1.0, 2.0, 0.4, ADD, -1.0, 3.0, False, {
          0.498689: 0.15865525,
      }),
      (4.0, 8.0, 0.2, ADD, -4.0, 12.0, False, {
          0.218575: 0.15865525,
      }),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, -2.0, 1.0, True, {
          math.inf: 0.15865525,
          -1.5: 0.02275013
      }),
      (3.0, 3.0, 1.0, REM, -6.0, 3.0, True, {
          math.inf: 0.15865525,
          -1.5: 0.02275013
      }),
      (1.0, 2.0, 1.0, REM, -3.0, 1.0, True, {
          math.inf: 0.15865525,
          -4.0: 0.00134989
      }),
      (4.0, 8.0, 1.0, REM, -12.0, 4.0, True, {
          math.inf: 0.15865525,
          -4.0: 0.00134989
      }),
      (1.0, 1.0, 1.0, REM, -2.0, 1.0, False, {
          1.5: 0.15865525,
      }),
      (3.0, 3.0, 1.0, REM, -6.0, 3.0, False, {
          1.5: 0.15865525,
      }),
      (1.0, 2.0, 1.0, REM, -3.0, 1.0, False, {
          4.0: 0.15865525,
      }),
      (4.0, 8.0, 1.0, REM, -12.0, 4.0, False, {
          4.0: 0.15865525,
      }),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, -2.0, 1.0, True, {
          math.inf: 0.1314742295348015,
          -0.971528299641668: 0.0499311563448348
      }),
      (3.0, 3.0, 0.8, REM, -6.0, 3.0, True, {
          math.inf: 0.1314742295348015,
          -0.971528299641668: 0.0499311563448348
      }),
      (1.0, 2.0, 0.5, REM, -3.0, 1.0, True, {
          math.inf: 0.0800025759815436,
          -0.6749972526421355: 0.0800025759815436,
      }),
      (4.0, 8.0, 0.6, REM, -12.0, 4.0, True, {
          math.inf: 0.0957331115715263,
          -0.88918789612552: 0.06427204039156087
      }),
      (1.0, 1.0, 0.9, REM, -2.0, 1.0, False, {
          1.419129383720773: 0.1450647417331293,
      }),
      (3.0, 3.0, 0.7, REM, -6.0, 3.0, False, {
          1.23465205122806: 0.1178837173364737,
      }),
      (1.0, 2.0, 0.4, REM, -3.0, 1.0, False, {
          3.110812103874479: 0.06427204039156088,
      }),
      (4.0, 8, 0.2, REM, -12.0, 4.0, False, {
          2.461265214250274: 0.03281096921159548,
      }))
  def test_gaussian_privacy_loss_tail(self, standard_deviation, sensitivity,
                                      sampling_prob, adjacency_type,
                                      expected_lower_x_truncation,
                                      expected_upper_x_truncation,
                                      pessimistic_estimate,
                                      expected_tail_probability_mass_function):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        pessimistic_estimate=pessimistic_estimate,
        log_mass_truncation_bound=math.log(2) + stats.norm.logcdf(-1),
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.assert_dictionary_almost_equal(
        self, expected_tail_probability_mass_function,
        tail_pld.tail_probability_mass_function)

  @parameterized.parameters((0.0, 1.0), (-10.0, 2.0), (4.0, 0.0), (2.0, -1.0),
                            (1.0, 1.0, 1.0, ADD, 1), (2.0, 1.0, 0.0, REM),
                            (1.0, 1.0, 1.2, ADD), (2.0, 1.0, -0.1, REM))
  def test_gaussian_value_errors(self, standard_deviation, sensitivity,
                                 sampling_prob=1.0, adjacency_type=ADD,
                                 log_mass_truncation_bound=-50):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.GaussianPrivacyLoss(
          standard_deviation,
          sensitivity=sensitivity,
          log_mass_truncation_bound=log_mass_truncation_bound,
          sampling_prob=sampling_prob,
          adjacency_type=adjacency_type)

  @parameterized.parameters((1.0, 1.0, 1.0, 0), (1.0, 1.0, 1.0, 1.1),
                            (1.0, 1.0, 1.0, -0.1), (1.0, 0, 1.0, 0.1),
                            (1.0, -0.2, 1.0, 0.1), (1.0, 1.1, 1.0, 0.2))
  def test_gaussian_from_privacy_parameters_value_errors(
      self, sensitivity, sampling_prob, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.GaussianPrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity,
          sampling_prob=sampling_prob)

  @parameterized.parameters((1.0, 1.0, ADD, 1.0, 0.12693674, 1.0),
                            (2.0, 1.0, REM, 1.0, 0.12693674, 2.0),
                            (3.0, 1.0, ADD, 1.0, 0.78760074, 1.0),
                            (6.0, 1.0, REM, 1.0, 0.78760074, 2.0),
                            (1.0, 1.0, ADD, 2.0, 0.02092364, 1.0),
                            (5.0, 1.0, REM, 2.0, 0.02092364, 5.0),
                            (1.0, 1.0, ADD, 16.0, 1e-5, 0.344),
                            (2.0, 1.0, REM, 16.0, 1e-5, 0.688),
                            (1.0, 0.8, ADD, 1.0, 0.081695179, 1.0),
                            (2.0, 0.7, ADD, 1.0, 0.143886147, 1.5),
                            (3.0, 0.5, ADD, 1.0, 0.267379199, 1.3),
                            (6.0, 0.01, ADD, 1.0, 0.0030216468, 2.0),
                            (1.0, 0.1, REM, 2.0, 2.355186318853955e-6, 1.0),
                            (5.0, 0.75, REM, 2.0, 0.0087720149, 5.0),
                            (1.0, 0.3, REM, 16, 0.0000329405, 0.3),
                            (2.0, 0.2, REM, 16, 0.0230238234, 0.4))
  def test_gaussian_from_privacy_parameters(self, sensitivity, sampling_prob,
                                            adjacency_type, epsilon, delta,
                                            expected_standard_deviation):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss.from_privacy_guarantee(
        common.DifferentialPrivacyParameters(epsilon, delta),
        sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_standard_deviation, pl.standard_deviation,
                           3)
    self.assertEqual(adjacency_type, pl.adjacency_type)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1.0, 1.0, ADD, 1.0, 0.12693674),
      (2.0, 2.0, 1.0, ADD, 1.0, 0.12693674),
      (1.0, 3.0, 1.0, ADD, 1.0, 0.78760074),
      (2.0, 6.0, 1.0, ADD, 1.0, 0.78760074),
      (1.0, 1.0, 1.0, ADD, 2.0, 0.02092364),
      (5.0, 5.0, 1.0, ADD, 2.0, 0.02092364),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1.0, 0.8, ADD, 1.0, 0.0231362104090899),
      (2.0, 2.0, 0.8, ADD, 1.0, 0.0231362104090899),
      (1.0, 3.0, 0.7, ADD, 1.0, 0.1195051215523554),
      (2.0, 6.0, 0.4, ADD, 1.0, 0.0),
      (1.0, 1.0, 0.3, ADD, 2.0, 0.0),
      (5.0, 5.0, 0.2, ADD, 2.0, 0.0),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1.0, 1.0, REM, 1.0, 0.12693674),
      (2.0, 2.0, 1.0, REM, 1.0, 0.12693674),
      (1.0, 3.0, 1.0, REM, 1.0, 0.78760074),
      (2.0, 6.0, 1.0, REM, 1.0, 0.78760074),
      (1.0, 1.0, 1.0, REM, 2.0, 0.02092364),
      (5.0, 5.0, 1.0, REM, 2.0, 0.02092364),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1.0, 0.8, REM, 1.0, 0.0816951786585355),
      (2.0, 2.0, 0.8, REM, 1.0, 0.0816951786585355),
      (1.0, 3.0, 0.7, REM, 1.0, 0.5356298793262404),
      (2.0, 6.0, 0.4, REM, 1.0, 0.2888308005139968),
      (1.0, 1.0, 0.3, REM, 2.0, 0.0003341102928869332),
      (5.0, 5.0, 0.2, REM, -0.25, 0.2211992169285951))
  def test_gaussian_get_delta_for_epsilon(
      self, standard_deviation, sensitivity, sampling_prob, adjacency_type,
      epsilon, expected_delta):
    pl = privacy_loss_mechanism.GaussianPrivacyLoss(
        standard_deviation,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_delta, pl.get_delta_for_epsilon(epsilon))


class DiscreteLaplacePrivacyLossDistributionTest(parameterized.TestCase):

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 0, 1.0),
      (1.0, 1, 1.0, ADD, 1, -1.0),
      (0.3, 2, 1.0, ADD, 0, 0.6),
      (0.3, 2, 1.0, ADD, 1, 0.0),
      (0.3, 2, 1.0, ADD, 2, -0.6),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 1, -0.86483972516319),
      (1.0, 1, 0.8, ADD, -1, 0.7046054708796525),
      (0.3, 2, 0.5, ADD, 2, -0.3443407699259402),
      (0.3, 3, 0.5, ADD, 2, -0.1612080639085818),
      (0.3, 2, 0.4, ADD, 1, 0),
      (0.3, 2, 0.3, ADD, 0, 0.1454380063386891),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, -1, 1.0),
      (1.0, 1, 1.0, REM, 0, -1.0),
      (0.3, 2, 1.0, REM, -2, 0.6),
      (0.3, 2, 1.0, REM, -1, 0),
      (0.3, 2, 1.0, REM, 0, -0.6),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, -1, 0.86483972516319),
      (1.0, 1, 0.8, REM, 1, -0.7046054708796525),
      (0.3, 2, 0.5, REM, -2, 0.3443407699259402),
      (0.3, 3, 0.5, REM, -2, 0.1612080639085818),
      (0.3, 2, 0.4, REM, -1, 0),
      (0.3, 2, 0.3, REM, 0, -0.1454380063386891))
  def test_discrete_laplace_privacy_loss(self, parameter, sensitivity,
                                         sampling_prob, adjacency_type, x,
                                         expected_privacy_loss):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters((1.0, 1, 0.4), (2.0, 7, -1.1))
  def test_discrete_laplace_privacy_loss_value_errors(
      self, parameter, sensitivity, x):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter, sensitivity=sensitivity)
    with self.assertRaises(ValueError):
      pl.privacy_loss(x)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 1.1, -math.inf),
      (1.0, 1, 1.0, ADD, 0.9, 0.0),
      (1.0, 1, 1.0, ADD, -1.0, math.inf),
      (0.3, 2, 1.0, ADD, 0.7, -math.inf),
      (0.3, 2, 1.0, ADD, 0.2, 0),
      (0.3, 2, 1.0, ADD, 0.0, 1.0),
      (0.3, 2, 1.0, ADD, -0.6, math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 0.9, -math.inf),
      (1.0, 1, 0.8, ADD, 0.7, 0),
      (1.0, 1, 0.8, ADD, -0.9, math.inf),
      (0.3, 2, 0.5, ADD, 0.26, -math.inf),
      (0.3, 2, 0.4, ADD, 0.0, 1.0),
      (0.3, 2, 0.3, ADD, -0.23, math.inf),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, 1.1, -math.inf),
      (1.0, 1, 1.0, REM, 0.9, -1.0),
      (1.0, 1, 1.0, REM, -1.0, math.inf),
      (0.3, 2, 1.0, REM, 0.7, -math.inf),
      (0.3, 2, 1.0, REM, 0.2, -2.0),
      (0.3, 2, 1.0, REM, 0.0, -1.0),
      (0.3, 2, 1.0, REM, -0.6, math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, 0.9, -math.inf),
      (1.0, 1, 0.8, REM, 0.86483972516319, -1.0),
      (1.0, 1, 0.8, REM, -0.8, math.inf),
      (0.3, 2, 0.5, REM, 0.35, -math.inf),
      (0.3, 2, 0.4, REM, 0.0, -1.0),
      (0.3, 2, 0.3, REM, -0.15, math.inf))
  def test_discrete_laplace_inverse_privacy_loss(self, parameter, sensitivity,
                                                 sampling_prob, adjacency_type,
                                                 privacy_loss, expected_x):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 1, 0, {
          1: 0.73105858,
          -1: 0.26894142
      }),
      (0.3, 2, 1.0, ADD, 1, 1, {
          0.6: 0.57444252,
          -0.6: 0.31526074
      }),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 1, 0, {
          0.7046054708796525: 0.73105858,
          -0.86483972516319: 0.26894142
      }),
      (0.3, 2, 0.6, ADD, 1, 1, {
          0.3156879596155301: 0.5744425168116589,
          -0.4009692034808894: 0.3152607374933769
      }),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, 0, -1, {
          1: 0.73105858,
          -1: 0.26894142
      }),
      (0.3, 2, 1.0, REM, -1, -1, {
          0.6: 0.57444252,
          -0.6: 0.31526074
      }),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, 0, -1, {
          0.86483972516319: 0.638635147178003,
          -0.7046054708796525: 0.361364852821997
      }),
      (0.3, 2, 0.6, REM, -1, -1, {
          0.4009692034808894: 0.4707698050843462,
          -0.3156879596155301: 0.4189334492206898
      }))
  def test_discrete_laplace_privacy_loss_tail(
      self, parameter, sensitivity, sampling_prob, adjacency_type,
      expected_lower_x_truncation, expected_upper_x_truncation,
      expected_tail_probability_mass_function):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.assert_dictionary_almost_equal(
        self, expected_tail_probability_mass_function,
        tail_pld.tail_probability_mass_function)

  @parameterized.parameters((-3.0, 1), (0.0, 1), (2.0, 0.5),
                            (2.0, -1), (1.0, 0),
                            (2.0, 1, 0.0, ADD), (1.0, 1, 1.2, REM),
                            (2.0, 1, -0.1, ADD))
  def test_discrete_laplace_value_errors(self, parameter, sensitivity,
                                         sampling_prob=1.0, adjacency_type=ADD):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
          parameter,
          sensitivity=sensitivity,
          sampling_prob=sampling_prob,
          adjacency_type=adjacency_type)

  @parameterized.parameters((-1, 0.8, 1.0, 0.1), (0.5, 1.0, 1.0, 0.1),
                            (0, 1.0, 1.0, 0.2), (1, 1.0, 1.0, -0.1),
                            (1, 0.8, 1.0, 1.1), (1, 0.0, 1.0, 0.1),
                            (3, 1.1, 1.0, 0.1), (1, -0.2, 1.0, 0.1))
  def test_discrete_laplace_from_privacy_parameters_value_errors(
      self, sensitivity, sampling_prob, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteLaplacePrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity, sampling_prob=sampling_prob)

  @parameterized.parameters((1, 1.0, ADD, 1.0, 0.0, 1.0),
                            (1, 1.0, REM, 1.0, 0.1, 1.0),
                            (2, 1.0, ADD, 1.0, 0.01, 0.5),
                            (1, 1.0, REM, 3.0, 0.01, 3.0),
                            (1, 0.8, ADD, 1.0, 0.0, 1.1467204062),
                            (1, 0.7, REM, 1.0, 0.1, 1.2397322437),
                            (2, 0.3, ADD, 1.0, 0.01, 0.9531096869),
                            (1, 0.2, REM, 3.0, 0.01, 4.5687933452))
  def test_discrete_laplace_from_privacy_parameters(
      self, sensitivity, sampling_prob, adjacency_type,
      epsilon, delta, expected_parameter):
    pl = (privacy_loss_mechanism.DiscreteLaplacePrivacyLoss
          .from_privacy_guarantee(
              common.DifferentialPrivacyParameters(
                  epsilon, delta),
              sensitivity,
              sampling_prob=sampling_prob,
              adjacency_type=adjacency_type))
    self.assertAlmostEqual(expected_parameter, pl.parameter)
    self.assertEqual(adjacency_type, pl.adjacency_type)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 1.0, 0.0), (0.333333, 3, 1.0, ADD, 1.0, 0.0),
      (0.5, 4, 1.0, ADD, 2.0, 0.0), (0.5, 4, 1.0, ADD, 0.5, 0.54202002),
      (0.5, 4, 1.0, ADD, 1.0, 0.39346934),
      (0.5, 4, 1.0, ADD, -0.5, 0.72222110),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 1.0, 0.0),
      (0.333333, 3, 0.8, ADD, 1.0, 0.0),
      (0.5, 4, 0.7, ADD, 0.5, 0.2293628348747755),
      (0.5, 4, 0.6, ADD, 0.6, 0.07668344250639381),
      (0.5, 4, 0.3, ADD, 0.5, 0.0),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, 1.0, 0.0), (0.333333, 3, 1.0, REM, 1.0, 0.0),
      (0.5, 4, 1.0, REM, 2.0, 0.0), (0.5, 4, 1.0, REM, 0.5, 0.54202002),
      (0.5, 4, 1.0, REM, 1.0, 0.39346934),
      (0.5, 4, 1.0, REM, -0.5, 0.72222110),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, 1.0, 0.0),
      (0.333333, 3, 0.8, REM, 1.0, 0.0),
      (0.5, 4, 0.7, REM, 0.5, 0.3523838505224567),
      (0.5, 4, 0.6, REM, 1.0, 0.178181891763215),
      (0.5, 4, 0.3, REM, 0.5, 0.1068168460276349),
      (1.0, 1, 0.2, REM, -0.25, 0.2211992169285951))
  def test_discrete_laplace_get_delta_for_epsilon(self, parameter, sensitivity,
                                                  sampling_prob, adjacency_type,
                                                  epsilon, expected_delta):
    pl = privacy_loss_mechanism.DiscreteLaplacePrivacyLoss(
        parameter,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_delta, pl.get_delta_for_epsilon(epsilon))


class DiscreteGaussianPrivacyLossTest(parameterized.TestCase):

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, 5, -4.5),
      (1, 1, 1, ADD, -3, 3.5),
      (1, 2, 1, ADD, 3, -4.0),
      (4.0, 4, 1.0, ADD, 20, -4.5),
      (5, 5, 1, ADD, -15, 3.5),
      (7.0, 14, 1.0, ADD, 21, -4.0),
      (1.0, 1, 1.0, ADD, -12, math.inf),
      (1.0, 1, 1.0, ADD, 13, -math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, -4, 1.565960898891332),
      (1.0, 1, 0.7, ADD, 4, -3.156183763141021),
      (1.0, 2, 0.4, ADD, -1, 0.4986891437585786),
      (4.0, 4, 0.3, ADD, -16, 0.3519252431310541),
      (5.0, 5, 0.4, ADD, 20, -2.628009438900115),
      (7.0, 14, 0.1, ADD, -7, 0.1033275126220077),
      (1.0, 1, 0.3, ADD, 13, -math.inf),
      (1.0, 1, 0.3, ADD, -12, 0.3566749439387324),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, 4, -4.5), (1, 1, 1, REM, -4, 3.5),
      (1.0, 2, 1.0, REM, 1, -4.0), (4, 4, 1, REM, 16, -4.5),
      (5.0, 5, 1.0, REM, -20, 3.5), (7, 14, 1, REM, 7, -4.0),
      (1.0, 1, 1.0, REM, -13, math.inf),
      (1.0, 1, 1.0, REM, 12, -math.inf),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, 4, -1.565960898891332),
      (1.0, 1, 0.7, REM, -4, 3.156183763141021),
      (1.0, 2, 0.4, REM, 1, -0.4986891437585786),
      (4.0, 4, 0.3, REM, 16, -0.3519252431310541),
      (5.0, 5, 0.4, REM, -20, 2.628009438900115),
      (7.0, 14, 0.1, REM, 7, -0.1033275126220077),
      (1.0, 1, 0.3, REM, -13, math.inf),
      (1.0, 1, 0.3, REM, 12, -0.3566749439387324))
  def test_discrete_gaussian_privacy_loss(self, sigma, sensitivity,
                                          sampling_prob, adjacency_type,
                                          x, expected_privacy_loss):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_privacy_loss, pl.privacy_loss(x))

  @parameterized.parameters((1.0, 1, 1.0, ADD, 0.4), (2.0, 7, 1.0, REM, -1.1),
                            (1.0, 1, 0.6, ADD, -13), (2.0, 1, 0.5, ADD, 26),
                            (1.0, 1, 0.6, REM, -14), (2.0, 1, 0.5, REM, 25))
  def test_discrete_gaussian_privacy_loss_value_errors(self, sigma, sensitivity,
                                                       sampling_prob,
                                                       adjacency_type, x):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    with self.assertRaises(ValueError):
      pl.privacy_loss(x)

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 1.0, ADD, -4.5, 5),
      (1.0, 1, 1.0, ADD, 3.5, -3),
      (1.0, 2, 1.0, ADD, -4.0, 3),
      (4.0, 4, 1.0, ADD, -4.51, 20),
      (5.0, 5, 1.0, ADD, 3.49, -15),
      (7.0, 14, 1.0, ADD, -4.0, 21),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 0.8, ADD, 1.565961, -5),
      (1.0, 1, 0.7, ADD, -3.156182, 3),
      (1.0, 2, 0.4, ADD, 0.4986892, -2),
      (4.0, 4, 0.3, ADD, 0.3519254, -17),
      (5.0, 5, 0.4, ADD, -2.6280094, 19),
      (7.0, 14, 0.1, ADD, 0.1033276, -8),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 1.0, REM, -4.5, 4),
      (1.0, 1, 1.0, REM, 3.5, -4),
      (1.0, 2, 1.0, REM, -4.0, 1),
      (4.0, 4, 1.0, REM, -4.51, 16),
      (5.0, 5, 1.0, REM, 3.49, -20),
      (7.0, 14, 1.0, REM, -4.0, 7),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 0.8, REM, -1.565961, 4),
      (1.0, 1, 0.7, REM, 3.156182, -4),
      (1.0, 2, 0.4, REM, -0.4986892, 1),
      (4.0, 4, 0.3, REM, -0.3519254, 16),
      (5.0, 5, 0.4, REM, 2.6280094, -20),
      (7.0, 14, 0.1, REM, -0.1033276, 7))
  def test_discrete_gaussian_inverse_privacy_loss(self, sigma, sensitivity,
                                                  sampling_prob, adjacency_type,
                                                  privacy_loss, expected_x):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    self.assertAlmostEqual(expected_x, pl.inverse_privacy_loss(privacy_loss))

  @parameterized.parameters(
      # Tests with sampling_prob = 1 for adjacency_type=ADD
      (1.0, 1, 2, 1.0, ADD, -1, 2, {
          math.inf: 0.05448868
      }),
      (1.0, 2, 2, 1.0, ADD, 0, 2, {
          math.inf: 0.29869003
      }),
      # Tests with sampling_prob < 1 for adjacency_type=ADD
      (1.0, 1, 2, 0.8, ADD, -2, 2, {
          math.inf: 0.0
      }),
      (1.0, 2, 2, 0.7, ADD, -2, 2, {
          math.inf: 0.0
      }),
      # Tests with sampling_prob = 1 for adjacency_type=REMOVE
      (1.0, 1, 2, 1.0, REM, -2, 1, {
          math.inf: 0.05448868
      }),
      (1.0, 2, 2, 1.0, REM, -2, 0, {
          math.inf: 0.29869003
      }),
      # Tests with sampling_prob < 1 for adjacency_type=REMOVE
      (1.0, 1, 2, 0.8, REM, -2, 2, {
          math.inf: 0.043590944
      }),
      (1.0, 2, 2, 0.7, REM, -2, 2, {
          math.inf: 0.209083021
      }))
  def test_discrete_gaussian_privacy_loss_tail(
      self, sigma, sensitivity, truncation_bound, sampling_prob, adjacency_type,
      expected_lower_x_truncation, expected_upper_x_truncation,
      expected_tail_probability_mass_function):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma,
        sensitivity=sensitivity,
        truncation_bound=truncation_bound,
        sampling_prob=sampling_prob,
        adjacency_type=adjacency_type)
    tail_pld = pl.privacy_loss_tail()
    self.assertAlmostEqual(expected_lower_x_truncation,
                           tail_pld.lower_x_truncation)
    self.assertAlmostEqual(expected_upper_x_truncation,
                           tail_pld.upper_x_truncation)
    test_util.assert_dictionary_almost_equal(
        self, expected_tail_probability_mass_function,
        tail_pld.tail_probability_mass_function)

  @parameterized.parameters((-3.0, 1), (0.0, 1), (2.0, 0.5), (1.0, 0),
                            (2.0, -1), (2.0, 4, 1, ADD, 1),
                            (2.0, 1, 0), (1.0, 1, 1.2), (2.0, 1, -0.1))
  def test_discrete_gaussian_value_errors(self, sigma, sensitivity,
                                          sampling_prob=1.0, adjacency_type=ADD,
                                          truncation_bound=None):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
          sigma,
          sensitivity=sensitivity,
          truncation_bound=truncation_bound,
          sampling_prob=sampling_prob,
          adjacency_type=adjacency_type)

  @parameterized.parameters((1.0, 1, 1, {
      -1.5: 0,
      -1: 0.27406862,
      0: 0.7259314,
      1: 1,
      1.5: 1
  }), (3.0, 2, 2, {
      -2.1: 0,
      -2: 0.17820326,
      -1: 0.38872553,
      0: 0.61127447,
      1: 0.82179674,
      2: 1,
      2.7: 1
  }))
  def test_discrete_gaussian_noise_cdf(self, sigma, sensitivity,
                                       truncation_bound, x_to_cdf_value):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma, sensitivity=sensitivity, truncation_bound=truncation_bound)
    for x, cdf_value in x_to_cdf_value.items():
      self.assertAlmostEqual(cdf_value, pl.noise_cdf(x))

  @parameterized.parameters((1.0, 1, 1, 0.7403629), (3.0, 2, 2, 1.3589226))
  def test_discrete_gaussian_std(self, sigma, sensitivity, truncation_bound,
                                 expected_std):
    pl = privacy_loss_mechanism.DiscreteGaussianPrivacyLoss(
        sigma, sensitivity=sensitivity, truncation_bound=truncation_bound)
    self.assertAlmostEqual(expected_std, pl.standard_deviation())

  @parameterized.parameters((-1, 1.0, 1.0, 0.1), (0.5, 1.0, 1.0, 0.1),
                            (0, 0.7, 1.0, 0.2), (1, 1.0, 1.0, 0.0),
                            (1, 1.0, 1.0, 1.1), (1, 1.0, 1.0, -0.1),
                            (1, 0.0, 1.0, 0.1), (1, 1.1, 1.0, 0.1),
                            (1, -0.1, 1.0, 0.1))
  def test_discrete_gaussian_from_privacy_parameters_value_errors(
      self, sensitivity, sampling_prob, epsilon, delta):
    with self.assertRaises(ValueError):
      privacy_loss_mechanism.DiscreteGaussianPrivacyLoss.from_privacy_guarantee(
          common.DifferentialPrivacyParameters(epsilon, delta),
          sensitivity,
          sampling_prob=sampling_prob)

  @parameterized.parameters(
      (1, 1.0, ADD, 1.0, 0.12693674, 1.041),
      (2, 1.0, REM, 1.0, 0.12693674, 1.972),
      (3, 1.0, ADD, 1.0, 0.78760074, 0.993),
      (6, 1.0, REM, 1.0, 0.78760074, 2.014),
      (1, 1.0, ADD, 2.0, 0.02092364, 1.038),
      (5, 1.0, REM, 2.0, 0.02092364, 5.008),
      (1, 1.0, ADD, 16.0, 1e-5, 0.306),
      (2, 1.0, REM, 16.0, 1e-5, 0.703),
      (1, 0.8, REM, 1.0, 0.07850075632001355, 1.041),
      (2, 0.7, ADD, 1.0, 0.06665777574091321, 1.972),
      (3, 0.4, REM, 1.0, 0.27122238416249084, 0.993),
      (6, 0.5, ADD, 1.0, 0.3604879495041193, 2.014),
      (1, 0.3, REM, 2.0, 0.0002834863230938751, 1.038),
      (5, 0.1, ADD, 2.0, 2.340272571167144e-06, 5.008),
      (2, 0.9, REM, 16.0, 4.518347272315105e-06, 0.703))
  def test_discrete_gaussian_from_privacy_parameters(self, sensitivity,
                                                     sampling_prob,
                                                     adjacency_type, epsilon,
                                                     delta, expected_sigma):
    pl = (
        privacy_loss_mechanism.DiscreteGaussianPrivacyLoss
        .from_privacy_guarantee(
            common.DifferentialPrivacyParameters(epsilon, delta),
            sensitivity,
            sampling_prob=sampling_prob,
            adjacency_type=adjacency_type))
    self.assertAlmostEqual(expected_sigma, pl._sigma, 3)
    self.assertEqual(adjacency_type, pl.adjacency_type)

if __name__ == '__main__':
  unittest.main()
