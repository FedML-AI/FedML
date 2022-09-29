"""Implementing Privacy Loss of Mechanisms.

This file implements privacy loss of several additive noise mechanisms,
including Gaussian Mechanism, Laplace Mechanism and Discrete Laplace Mechanism.
Please refer to the supplementary material below for more details:
../docs/Privacy_Loss_Distributions.pdf
"""

import abc
import dataclasses
import enum
import math
from typing import Iterable, Mapping, Optional, Union
import numpy as np
from scipy import stats

from fedml.core.dp.budget_accountant import common


class AdjacencyType(enum.Enum):
    """Designates the type of adjacency for computing privacy loss distributions.

    ADD: the 'add' adjacency type specifies that the privacy loss distribution
      for a mechanism M is to be computed with mu_upper = M(D) and mu_lower =
      M(D'), where D' contains one more datapoint than D.
    REMOVE: the 'remove' adjacency type specifies that the privacy loss
      distribution for a mechanism M is to be computed with mu_upper = M(D) and
      mu_lower = M(D'), where D' contains one less datapoint than D.

    Note: The rest of code currently assumes existence of only these two adjacency
    types. If a new adjacency type is added and used, the API in this file will
    pretend that it is same as REMOVE.
    """
    ADD = 'ADD'
    REMOVE = 'REMOVE'


@dataclasses.dataclass
class TailPrivacyLossDistribution(object):
    """Representation of the tail of privacy loss distribution.

    Attributes:
      lower_x_truncation: the minimum value of x that should be considered after
        the tail is discarded.
      upper_x_truncation: the maximum value of x that should be considered after
        the tail is discarded.
      tail_probability_mass_function: the probability mass of the privacy loss
        distribution that has to be added due to the discarded tail; each key is a
        privacy loss value and the corresponding value is the probability mass
        that the value occurs.
    """
    lower_x_truncation: float
    upper_x_truncation: float
    tail_probability_mass_function: Mapping[float, float]


class AdditiveNoisePrivacyLoss(metaclass=abc.ABCMeta):
    """Superclass for privacy loss of additive noise mechanisms.

    An additive noise mechanism for computing a scalar-valued function f is a
    mechanism that outputs the sum of the true value of the function and a noise
    drawn from a certain distribution mu. This class allows one to compute several
    quantities related to the privacy loss of additive noise mechanisms.

    We assume that the noise mu is such that the algorithm is more private as the
    sensitivity of f decreases. (Recall that the sensitivity of f is the maximum
    absolute change in f when an input to a single user changes.) Under this
    assumption, the privacy loss distribution of the mechanism is exactly
    generated as follows:
    - Let mu_lower(x) := mu(x - sensitivity), i.e., right shifted by sensitivity
    - Sample x ~ mu_upper = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).
    When mu is discrete, mu(x) refers to the probability mass of mu at x, and when
    mu is continuous, mu(x) is the probability density of mu at x; mu_upper and
    mu_lower are defined analogously.

    Support for sub-sampling (Refer to supplementary material for more details):
    An additive noise mechanism with Poisson sub-sampling first samples a subset
    of data points including each data point independently with probability q,
    and outputs the sum of the true value of the function and a noise drawn from
    a certain distribution mu. Here, we consider differential privacy with
    respect to the addition/removal relation.

    With sub-sampling probability of q, the privacy loss distribution is
    generated as follows:
    For ADD adjacency type:
    - Let mu_lower(x) := q * mu(x - sensitivity) + (1-q) * mu(x)
    - Sample x ~ mu_upper = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).
    For REMOVE adjacency type:
    - Let mu_upper(x) := q * mu(x + sensitivity) + (1-q) * mu(x)
    - Sample x ~ mu_lower = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).

    Note: When q = 1, the result privacy loss distributions for both ADD and
      REMOVE adjacency types are identical.

    This class also assumes the privacy loss is non-increasing as x increases.

    Attributes:
      sensitivity: the sensitivity of function f. (i.e. the maximum absolute
        change in f when an input to a single user changes.)
      discrete_noise: a value indicating whether the noise is discrete. If this
          is True, then it is assumed that the noise can only take integer values.
          If False, then it is assumed that the noise is continuous, i.e., the
          probability mass at any given point is zero.
      sampling_prob: sub-sampling probability, a value in (0,1].
      adjacency_type: type of adjacency relation to used for defining the privacy
          loss distribution.
    """

    def __init__(self,
                 sensitivity: float,
                 discrete_noise: bool,
                 sampling_prob: float = 1.0,
                 adjacency_type: AdjacencyType = AdjacencyType.REMOVE):
        if sensitivity <= 0:
            raise ValueError(
                f'Sensitivity is not a positive real number: {sensitivity}')
        if sampling_prob <= 0 or sampling_prob > 1:
            raise ValueError(
                f'Sampling probability is not in (0,1] : {sampling_prob}')
        self.sensitivity = sensitivity
        self.discrete_noise = discrete_noise
        self.sampling_prob = sampling_prob
        self.adjacency_type = adjacency_type

    def mu_upper_cdf(
            self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        """Computes the cumulative density function of the mu_upper distribution.

        For ADD adjacency type, for any sub-sampling probability:
          mu_upper(x) := mu
        For REMOVE adjacency type, with sub-sampling probability q:
          mu_upper(x) := (1-q) * mu(x) + q * mu(x + sensitivity)

        Args:
          x: the point or points at which the cumulative density function is to be
            calculated.

        Returns:
          The cumulative density function of the mu_upper distribution at x, i.e.,
          the probability that mu_upper is less than or equal to x.
        """

        if self.adjacency_type == AdjacencyType.ADD:
            return self.noise_cdf(x)
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            # For performance, the case of sampling_prob=1 is handled separately.
            if self.sampling_prob == 1.0:
                return self.noise_cdf(np.add(x, self.sensitivity))
            return ((1 - self.sampling_prob) * self.noise_cdf(x) +
                    self.sampling_prob * self.noise_cdf(np.add(x, self.sensitivity)))

    def mu_lower_cdf(
            self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        """Computes the cumulative density function of the mu_lower distribution.

        For ADD adjacency type, with sub-sampling probability q:
          mu_lower(x) := (1-q) * mu(x) + q * mu(x - sensitivity)
        For REMOVE adjacency type, for any sub-sampling probability:
          mu_lower(x) := mu(x)

        Args:
          x: the point or points at which the cumulative density function is to be
            calculated.

        Returns:
          The cumulative density function of the mu_lower distribution at x, i.e.,
          the probability that mu_lower is less than or equal to x.
        """
        if self.adjacency_type == AdjacencyType.ADD:
            # For performance, the case of sampling_prob=1 is handled separately.
            if self.sampling_prob == 1.0:
                return self.noise_cdf(np.add(x, -self.sensitivity))
            return ((1 - self.sampling_prob) * self.noise_cdf(x) +
                    self.sampling_prob * self.noise_cdf(np.add(x, -self.sensitivity)))

        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return self.noise_cdf(x)

    def get_delta_for_epsilon(self, epsilon):
        """Computes the epsilon-hockey stick divergence of the mechanism.

        The epsilon-hockey stick divergence of the mechanism is the value of delta
        for which the mechanism is (epsilon, delta)-differentially private. (See
        Observation 1 in the supplementary material.)

        This function assumes the privacy loss is non-increasing as x increases.
        Under this assumption, the hockey stick divergence is simply
        mu_upper_cdf(inverse_privacy_loss(epsilon)) - exp(epsilon) *
        mu_lower_cdf(inverse_privacy_loss(epsilon) - sensitivity), because the
        privacy loss at a point x is at least epsilon iff
        x <= inverse_privacy_loss(epsilon).

        When adjacency_type is ADD and epsilon >= -log(1 - sampling_prob),
          the hockey stick divergence is 0,
          since mu_lower_cdf*exp(epsilon) is pointwise greater than mu_upper_cdf.
        When adjacency_type is REMOVE and epsilon <= log(1 - sampling_prob),
          the hockey stick divergence is 1-exp(epsilon),
          since mu_lower_cdf*exp(epsilon) is pointwise lower than mu_upper_cdf.

        Args:
          epsilon: the epsilon in epsilon-hockey stick divergence.

        Returns:
          A non-negative real number which is the epsilon-hockey stick divergence
          of the mechanism.
        """
        if self.sampling_prob != 1.0:
            if (self.adjacency_type == AdjacencyType.ADD and
                    epsilon >= -math.log(1 - self.sampling_prob)):
                return 0.0
            if (self.adjacency_type == AdjacencyType.REMOVE and
                    epsilon <= math.log(1 - self.sampling_prob)):
                return 1.0 - math.exp(epsilon)
        x_cutoff = self.inverse_privacy_loss(epsilon)
        return (self.mu_upper_cdf(x_cutoff) -
                math.exp(epsilon) * self.mu_lower_cdf(x_cutoff))

    @abc.abstractmethod
    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        """Computes the privacy loss at the tail of the distribution.

        Returns:
          A TailPrivacyLossDistribution instance representing the tail of the
          privacy loss distribution.

        Raises:
          NotImplementedError: If not implemented by the subclass.
        """
        raise NotImplementedError

    def privacy_loss(self, x: float) -> float:
        """Computes the privacy loss at a given point.

        For ADD adjacency type, with sub-sampling probability of q:
        the privacy loss at x is
        - log(1-q + q*exp(-privacy_loss_without_subsampling(x))).

        For REMOVE adjacency type, with sub-sampling probability of q:
        the privacy loss at x is
        log(1-q + q*exp(privacy_loss_without_subsampling(x))).

        Args:
          x: the point at which the privacy loss is computed.

        Returns:
          The privacy loss at point x.

        Raises:
          NotImplementedError: If privacy_loss_without_subsampling is not
            implemented by the subclass.
          ValueError: If privacy loss is undefined at x.
        """
        privacy_loss_without_subsampling = self.privacy_loss_without_subsampling(x)
        # For performance, the case of sampling_prob=1 is handled separately.
        if self.sampling_prob == 1.0:
            return privacy_loss_without_subsampling
        if self.adjacency_type == AdjacencyType.ADD:
            return -math.log(1 - self.sampling_prob + self.sampling_prob *
                             math.exp(-privacy_loss_without_subsampling))
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return math.log(1 - self.sampling_prob + self.sampling_prob *
                            math.exp(privacy_loss_without_subsampling))

    @abc.abstractmethod
    def privacy_loss_without_subsampling(self, x: float) -> float:
        """Computes the privacy loss at a given point without sub-sampling.

        Args:
          x: the point at which the privacy loss is computed.

        Returns:
          The privacy loss at point x without sub-sampling, which is given as:
          For ADD adjacency type: ln(mu(x - sensitivity) / mu(x)).
          If mu(x - sensitivity) == 0 and mu(x) > 0, this is -infinity.
          If mu(x - sensitivity) > 0  and mu(x) == 0, this is +infinity.
          If mu(x - sensitivity) == 0 and mu(x) == 0, this is undefined
            (ValueError is raised in this case).

          For REMOVE adjacency type: ln(mu(x + sensitivity) / mu(x)).
          Similar conventions (regarding corner cases) apply as above.

        Raises:
          NotImplementedError: If not implemented by the subclass.
        """
        raise NotImplementedError

    def inverse_privacy_loss(self, privacy_loss: float) -> float:
        """Computes the inverse of a given privacy loss.

        Args:
          privacy_loss: the privacy loss value.

        Returns:
          The largest float x such that the privacy loss at x is at least
          privacy_loss.

          For the ADD adjacency type, with sub-sampling probability of q:
          the inverse privacy loss is given as
           inverse_privacy_loss_without_subsampling(-log(1 +
                                                         (exp(-privacy_loss)-1)/q)),
          When privacy_loss >= -log(1-q), the inverse privacy loss is
            inverse_privacy_loss_without_subsampling(+infinity),
          When privacy_loss == -infinity, the inverse privacy loss is
            inverse_privacy_loss_without_subsampling(-infinity).

          For the REMOVE adjacency type, with sub-sampling probability of q:
          the inverse privacy loss is given as
            inverse_privacy_loss_without_subsampling(log(1 +
                                                         (exp(privacy_loss)-1)/q)),
          When privacy_loss <= log(1-q), the inverse privacy loss is
            inverse_privacy_loss_without_subsampling(-infinity),
          When privacy_loss == infinity, the inverse privacy loss is
            inverse_privacy_loss_without_subsampling(+infinity).

        Raises:
          NotImplementedError: If inverse_privacy_loss_without_subsampling is not
            implemented by the subclass.
          ValueError: If inverse_privacy_loss_without_subsampling raises a
            ValueError
        """
        # For performance, the case of sampling_prob=1 is handled separately.
        if self.sampling_prob == 1.0:
            return self.inverse_privacy_loss_without_subsampling(privacy_loss)

        if self.adjacency_type == AdjacencyType.ADD:
            if math.isclose(privacy_loss, - math.log(1 - self.sampling_prob)):
                return self.inverse_privacy_loss_without_subsampling(math.inf)
            if privacy_loss > - math.log(1 - self.sampling_prob):
                raise ValueError(f'privacy_loss ({privacy_loss}) is larger than '
                                 f'-log(1 - sampling_prob) '
                                 f'({-math.log(1 - self.sampling_prob)}')
            return self.inverse_privacy_loss_without_subsampling(
                -math.log(1 + (math.exp(-privacy_loss) - 1) / self.sampling_prob))

        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            if math.isclose(privacy_loss, math.log(1 - self.sampling_prob)):
                return self.inverse_privacy_loss_without_subsampling(-math.inf)
            if privacy_loss <= math.log(1 - self.sampling_prob):
                raise ValueError(f'privacy_loss ({privacy_loss}) is smaller than '
                                 f'log(1 - sampling_prob) '
                                 f'({math.log(1 - self.sampling_prob)}')
            return self.inverse_privacy_loss_without_subsampling(
                math.log(1 + (math.exp(privacy_loss) - 1) / self.sampling_prob))

    @abc.abstractmethod
    def inverse_privacy_loss_without_subsampling(self,
                                                 privacy_loss: float) -> float:
        """Computes the inverse of a given privacy loss without sub-sampling.

        Args:
          privacy_loss: the privacy loss value.

        Returns:
          The largest float x such that the privacy loss at x without sub-sampling,
          is at least privacy_loss.

        Raises:
          NotImplementedError: If not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def noise_cdf(self, x: Union[float,
                                 Iterable[float]]) -> Union[float, np.ndarray]:
        """Computes the cumulative density function of the noise distribution mu.

        Args:
          x: the point or points at which the cumulative density function is to be
            calculated.

        Returns:
          The cumulative density function of that noise at x, i.e., the probability
          that mu is less than or equal to x.

        Raises:
          NotImplementedError: If not implemented by the subclass.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_privacy_guarantee(
            cls,
            privacy_parameters: common.DifferentialPrivacyParameters,
            sensitivity: float = 1,
            pessimistic_estimate: bool = True,
            sampling_prob: float = 1.0,
            adjacency_type: AdjacencyType = AdjacencyType.REMOVE
    ) -> 'AdditiveNoisePrivacyLoss':
        """Creates the privacy loss for the mechanism with a given privacy.

        Computes parameters achieving given privacy with REMOVE relation,
        irrespective of adjacency_type, since for all epsilon > 0, the hockey-stick
        divergence for PLD with respect to the REMOVE adjacency type is at least
        that for PLD with respect to ADD adjacency type.

        The returned object has the specified adjacency_type.

        Args:
          privacy_parameters: the desired privacy guarantee of the mechanism.
          sensitivity: the sensitivity of function f. (i.e. the maximum absolute
            change in f when an input to a single user changes.)
          pessimistic_estimate: a value indicating whether the rounding is done in
            such a way that the resulting epsilon-hockey stick divergence
            computation gives an upper estimate to the real value.
          sampling_prob: sub-sampling probability, a value in (0,1].
          adjacency_type: type of adjacency relation to used for defining the
            privacy loss distribution.

        Returns:
          The privacy loss of the mechanism with the given privacy guarantee.

        Raises:
          NotImplementedError: If not implemented by the subclass.
        """
        raise NotImplementedError


class LaplacePrivacyLoss(AdditiveNoisePrivacyLoss):
    """Privacy loss of the Laplace mechanism.

    The Laplace mechanism for computing a scalar-valued function f simply outputs
    the sum of the true value of the function and a noise drawn from the Laplace
    distribution. Recall that the Laplace distribution with parameter b has
    probability density function 0.5/b * exp(-|x|/b) at x for any real number x.

    The privacy loss distribution of the Laplace mechanism is equivalent to the
    privacy loss distribution between the Laplace distribution and the same
    distribution but shifted by the sensitivity of f. Specifically, the privacy
    loss distribution of the Laplace mechanism is generated as follows:
    - Let mu = Lap(0, b) be the Laplace noise PDF as given above.
    - Let mu_lower(x) := mu(x - sensitivity), i.e., right shifted by sensitivity
    - Sample x ~ mu_upper = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)), which is equal to
      (|x - sensitivity| - |x|) / parameter.

    Case of sub-sampling (Refer to supplementary material for more details):
    The Laplace mechanism with sub-sampling for computing a scalar-valued function
    f, first samples a subset of data points including each data point
    independently with probability q, and returns the sum of the true values and a
    noise drawn from the Laplace distribution. Here, we consider differential
    privacy with respect to the addition/removal relation.

    When the sub-sampling probability is q, the worst-case privacy loss
    distribution is generated as follows:
    For ADD adjacency type:
    - Let mu_lower(x) := q * mu(x - sensitivity) + (1-q) * mu(x)
    - Sample x ~ mu_upper = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).
    For REMOVE adjacency type:
    - Let mu_upper(x) := q * mu(x + sensitivity) + (1-q) * mu(x)
    - Sample x ~ mu_lower = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).

    Note: When q = 1, the result privacy loss distributions for both ADD and
      REMOVE adjacency types are identical.
    """

    def __init__(self,
                 parameter: float,
                 sensitivity: float = 1,
                 sampling_prob: float = 1.0,
                 adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> None:
        """Initializes the privacy loss of the Laplace mechanism.

        Args:
          parameter: the parameter of the Laplace distribution.
          sensitivity: the sensitivity of function f. (i.e. the maximum absolute
            change in f when an input to a single user changes.)
          sampling_prob: sub-sampling probability, a value in (0,1].
          adjacency_type: type of adjacency relation to used for defining the
            privacy loss distribution.
        """
        if parameter <= 0:
            raise ValueError(f'Parameter is not a positive real number: {parameter}')

        self._parameter = parameter
        self._laplace_random_variable = stats.laplace(scale=parameter)
        super().__init__(sensitivity, False, sampling_prob, adjacency_type)

    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        """Computes the privacy loss at the tail of the Laplace distribution.

        For ADD adjacency type:
        lower_x_truncation = 0 and upper_x_truncation = sensitivity

        For REMOVE adjacency type:
        lower_x_truncation = -sensitivity and upper_x_truncation = 0

        The probability masses below lower_x_truncation and above upper_x_truncation
        are computed using mu_upper_cdf.

        Returns:
          A TailPrivacyLossDistribution instance representing the tail of the
          privacy loss distribution.
        """
        if self.adjacency_type == AdjacencyType.ADD:
            lower_x_truncation, upper_x_truncation = 0.0, self.sensitivity
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            lower_x_truncation, upper_x_truncation = -self.sensitivity, 0.0

        return TailPrivacyLossDistribution(
            lower_x_truncation, upper_x_truncation, {
                self.privacy_loss(lower_x_truncation):
                    self.mu_upper_cdf(lower_x_truncation),
                self.privacy_loss(upper_x_truncation):
                    1 - self.mu_upper_cdf(upper_x_truncation)
            })

    def privacy_loss_without_subsampling(self, x: float) -> float:
        """Computes the privacy loss of the Laplace mechanism without sub-sampling at a given point.

        Args:
          x: the point at which the privacy loss is computed.

        Returns:
          The privacy loss of the Laplace mechanism without sub-sampling at point x,
          which is given as
          For ADD adjacency type:    (|x - sensitivity| - |x|) / parameter.
          For REMOVE adjacency type: (|x| - |x + sensitivity|) / parameter.
        """
        if self.adjacency_type == AdjacencyType.ADD:
            return (abs(x - self.sensitivity) - abs(x)) / self._parameter
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return (abs(x) - abs(x + self.sensitivity)) / self._parameter

    def inverse_privacy_loss_without_subsampling(self,
                                                 privacy_loss: float) -> float:
        """Computes the inverse of a given privacy loss for the Laplace mechanism without sub-sampling.

        Args:
          privacy_loss: the privacy loss value.

        Returns:
          The largest float x such that the privacy loss at x is at least
          privacy_loss.
          For ADD adjacency type:
            If privacy_loss <= - sensitivity / parameter, x is equal to infinity.
            If - sensitivity / parameter < privacy_loss <= sensitivity / parameter,
              x is equal to 0.5 * (sensitivity - privacy_loss * parameter).
            If privacy_loss > sensitivity / parameter, no such x exists and the
              function returns -infinity.
          For REMOVE adjacency type:
            For any value of privacy_loss, x is equal to the corresponding value for
              ADD adjacency type decreased by sensitivity.
        """
        loss_threshold = privacy_loss * self._parameter
        if loss_threshold > self.sensitivity:
            return -math.inf
        if loss_threshold <= -self.sensitivity:
            return math.inf
        if self.adjacency_type == AdjacencyType.ADD:
            return 0.5 * (self.sensitivity - loss_threshold)
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return 0.5 * (-self.sensitivity - loss_threshold)

    def noise_cdf(self, x: Union[float,
                                 Iterable[float]]) -> Union[float, np.ndarray]:
        """Computes the cumulative density function of the Laplace distribution.

        Args:
          x: the point or points at which the cumulative density function is to be
            calculated.

        Returns:
          The cumulative density function of the Laplace noise at x, i.e., the
          probability that the Laplace noise is less than or equal to x.
        """
        return self._laplace_random_variable.cdf(x)

    @classmethod
    def from_privacy_guarantee(
            cls,
            privacy_parameters: common.DifferentialPrivacyParameters,
            sensitivity: float = 1,
            pessimistic_estimate: bool = True,
            sampling_prob: float = 1.0,
            adjacency_type: AdjacencyType = AdjacencyType.REMOVE
    ) -> 'LaplacePrivacyLoss':
        """Creates the privacy loss for Laplace mechanism with given privacy.

        Without sub-sampling, the parameter of the Laplace mechanism is simply
          sensitivity / epsilon.
        With sub-sampling probability of q, the parameter is given as
          sensitivity / log(1 + (exp(epsilon) - 1)/q).
        Note: Only the REMOVE adjacency type is used in determining the parameter,
          since for all epsilon > 0, the hockey-stick divergence for PLD with
          respect to the REMOVE adjacency type is at least that for PLD with respect
          to ADD adjacency type.

        Args:
          privacy_parameters: the desired privacy guarantee of the mechanism.
          sensitivity: the sensitivity of function f. (i.e. the maximum absolute
            change in f when an input to a single user changes.)
          pessimistic_estimate: a value indicating whether the rounding is done in
            such a way that the resulting epsilon-hockey stick divergence
            computation gives an upper estimate to the real value.
          sampling_prob: sub-sampling probability, a value in (0,1].
          adjacency_type: type of adjacency relation to used for defining the
            privacy loss distribution.

        Returns:
          The privacy loss of the Laplace mechanism with the given privacy
            guarantee.
        """
        parameter = (
                sensitivity /
                np.log(1 + (np.exp(privacy_parameters.epsilon) - 1) / sampling_prob))
        return LaplacePrivacyLoss(
            parameter,
            sensitivity=sensitivity,
            sampling_prob=sampling_prob,
            adjacency_type=adjacency_type)

    @property
    def parameter(self) -> float:
        """The parameter of the corresponding Laplace noise."""
        return self._parameter


class GaussianPrivacyLoss(AdditiveNoisePrivacyLoss):
    """Privacy loss of the Gaussian mechanism.

    The Gaussian mechanism for computing a scalar-valued function f simply
    outputs the sum of the true value of the function and a noise drawn from the
    Gaussian distribution. Recall that the (centered) Gaussian distribution with
    standard deviation sigma has probability density function
    1/(sigma * sqrt(2 * pi)) * exp(-0.5 x^2/sigma^2) at x for any real number x.

    The privacy loss distribution of the Gaussian mechanism is equivalent to the
    privacy loss distribution between the Gaussian distribution and the same
    distribution but shifted by the sensitivity of f. Specifically, the privacy
    loss distribution of the Gaussian mechanism is generated as follows:
    - Let mu = N(0, sigma^2) be the Gaussian noise PDF as given above.
    - Let mu_lower(x) := mu(x - sensitivity), i.e., right shifted by sensitivity
    - Sample x ~ mu_upper = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).

    Case of sub-sampling (Refer to supplementary material for more details):
    The Gaussian mechanism with sub-sampling for computing a scalar-valued
    function f, first samples a subset of data points including each data point
    independently with probability q, and returns the sum of the true values and a
    noise drawn from the Gaussian distribution. Here, we consider differential
    privacy with respect to the addition/removal relation.

    When the sub-sampling probability is q, the worst-case privacy loss
    distribution is generated as follows:
    For ADD adjacency type:
    - Let mu_lower(x) := q * mu(x - sensitivity) + (1-q) * mu(x)
    - Sample x ~ mu_upper = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).
    For REMOVE adjacency type:
    - Let mu_upper(x) := q * mu(x + sensitivity) + (1-q) * mu(x)
    - Sample x ~ mu_lower = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).

    Note: When q = 1, the result privacy loss distributions for both ADD and
      REMOVE adjacency types are identical.
    """

    def __init__(self,
                 standard_deviation: float,
                 sensitivity: float = 1,
                 pessimistic_estimate: bool = True,
                 log_mass_truncation_bound: float = -50,
                 sampling_prob: float = 1.0,
                 adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> None:
        """Initializes the privacy loss of the Gaussian mechanism.

        Args:
          standard_deviation: the standard_deviation of the Gaussian distribution.
          sensitivity: the sensitivity of function f. (i.e. the maximum absolute
            change in f when an input to a single user changes.)
          pessimistic_estimate: a value indicating whether the rounding is done in
            such a way that the resulting epsilon-hockey stick divergence
            computation gives an upper estimate to the real value.
          log_mass_truncation_bound: the ln of the probability mass that might be
            discarded from the noise distribution. The larger this number, the more
            error it may introduce in divergence calculations.
          sampling_prob: sub-sampling probability, a value in (0,1].
          adjacency_type: type of adjacency relation to used for defining the
            privacy loss distribution.
        """
        if standard_deviation <= 0:
            raise ValueError(f'Standard deviation is not a positive real number: '
                             f'{standard_deviation}')
        if log_mass_truncation_bound > 0:
            raise ValueError(f'Log mass truncation bound is not a non-positive real '
                             f'number: {log_mass_truncation_bound}')

        self._standard_deviation = standard_deviation
        self._gaussian_random_variable = stats.norm(scale=standard_deviation)
        self._pessimistic_estimate = pessimistic_estimate
        self._log_mass_truncation_bound = log_mass_truncation_bound
        super().__init__(sensitivity, False, sampling_prob, adjacency_type)

    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        """Computes the privacy loss at the tail of the Gaussian distribution.

        For REMOVE adjacency type: lower_x_truncation is set such that
          CDF(lower_x_truncation) = 0.5 * exp(log_mass_truncation_bound), and
          upper_x_truncation is set to be -lower_x_truncation. Finally,
          lower_x_truncation is shifted by -1 * sensitivity.
          Recall that here mu_upper(x) := (1-q).mu(x) + q.mu(x + sensitivity),
          where q=sampling_prob. The truncations chosen above ensure that the tails
          of both mu(x) and mu(x+sensitivity) are smaller than 0.5 *
          exp(log_mass_truncation_bound). This ensures that the considered tails of
          mu_upper are no larger than exp(log_mass_truncation_bound). This is
          computationally cheaper than computing exact tail thresholds for mu_upper.

        For ADD adjacency type: lower_x_truncation is set such that
          CDF(lower_x_truncation) = 0.5 * exp(log_mass_truncation_bound), and
          upper_x_truncation is set to be -lower_x_truncation. Finally,
          upper_x_truncation is shifted by +1 * sensitivity.
          Recall that here mu_upper(x) := mu(x) for any value of sampling_prob.
          The truncations chosen ensures that the tails of mu(x) (and hence of
          mu_upper) are no larger than 0.5 * exp(log_mass_truncation_bound).
          While it was not strictly necessary to shift upper_x_truncation by +1 *
          sensitivity in this case, this choice leads to the same discretized
          privacy loss distribution for both ADD and REMOVE adjacency
          types, in the case where sampling_prob = 1.

        If pessimistic_estimate is True, the privacy losses for
        x < lower_x_truncation and x > upper_x_truncation are rounded up and added
        to tail_probability_mass_function. In the case x < lower_x_truncation,
        the privacy loss is rounded up to infinity. In the case
        x > upper_x_truncation, it is rounded up to the privacy loss at
        upper_x_truncation.

        On the other hand, if pessimistic_estimate is False, the privacy losses for
        x < lower_x_truncation and x > upper_x_truncation are rounded down and added
        to tail_probability_mass_function. In the case x < lower_x_truncation, the
        privacy loss is rounded down to the privacy loss at lower_x_truncation.
        In the case x > upper_x_truncation, it is rounded down to -infinity and
        hence not included in tail_probability_mass_function,

        Returns:
          A TailPrivacyLossDistribution instance representing the tail of the
          privacy loss distribution.
        """
        lower_x_truncation = self._gaussian_random_variable.ppf(
            0.5 * math.exp(self._log_mass_truncation_bound))
        upper_x_truncation = -lower_x_truncation
        if self.adjacency_type == AdjacencyType.ADD:
            upper_x_truncation += self.sensitivity
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            lower_x_truncation -= self.sensitivity
        if self._pessimistic_estimate:
            tail_probability_mass_function = {
                math.inf:
                    self.mu_upper_cdf(lower_x_truncation),
                self.privacy_loss(upper_x_truncation):
                    1 - self.mu_upper_cdf(upper_x_truncation)
            }
        else:
            tail_probability_mass_function = {
                self.privacy_loss(lower_x_truncation):
                    self.mu_upper_cdf(lower_x_truncation),
            }
        return TailPrivacyLossDistribution(lower_x_truncation, upper_x_truncation,
                                           tail_probability_mass_function)

    def privacy_loss_without_subsampling(self, x: float) -> float:
        """Computes the privacy loss of the Gaussian mechanism without sub-sampling at a given point.

        Args:
          x: the point at which the privacy loss is computed.

        Returns:
          The privacy loss of the Laplace mechanism at point x, which is given as
          For ADD adjacency type: (|x - sensitivity| - |x|) / parameter.
          For REMOVE adjacency type: (|x| - |x + sensitivity|) / parameter.
          The privacy loss of the Gaussian mechanism without sub-sampling at point
          x, which is given as
          For ADD adjacency type:
            sensitivity * (0.5 * sensitivity - x) / standard_deviation^2.
          For REMOVE adjacency type:
            sensitivity * (- 0.5 * sensitivity - x) / standard_deviation^2.
        """
        if self.adjacency_type == AdjacencyType.ADD:
            return (self.sensitivity * (0.5 * self.sensitivity - x) /
                    (self._standard_deviation ** 2))
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return (self.sensitivity * (-0.5 * self.sensitivity - x) /
                    (self._standard_deviation ** 2))

    def inverse_privacy_loss_without_subsampling(self,
                                                 privacy_loss: float) -> float:
        """Computes the inverse of a given privacy loss for the Gaussian mechanism without sub-sampling.

        Args:
          privacy_loss: the privacy loss value.

        Returns:
          The largest float x such that the privacy loss at x is at least
          privacy_loss. This is equal to
          For ADD adjacency type:
            0.5 * sensitivity - privacy_loss * standard_deviation^2 / sensitivity.
          For REMOVE adjacency type:
            -0.5 * sensitivity - privacy_loss * standard_deviation^2 / sensitivity.
        """
        if self.adjacency_type == AdjacencyType.ADD:
            return (0.5 * self.sensitivity - privacy_loss *
                    (self._standard_deviation ** 2) / self.sensitivity)
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return (-0.5 * self.sensitivity - privacy_loss *
                    (self._standard_deviation ** 2) / self.sensitivity)

    def noise_cdf(self, x: Union[float,
                                 Iterable[float]]) -> Union[float, np.ndarray]:
        """Computes the cumulative density function of the Gaussian distribution.

        Args:
         x: the point or points at which the cumulative density function is to be
           calculated.

        Returns:
          The cumulative density function of the Gaussian noise at x, i.e., the
          probability that the Gaussian noise is less than or equal to x.
        """
        return self._gaussian_random_variable.cdf(x)

    @classmethod
    def from_privacy_guarantee(
            cls,
            privacy_parameters: common.DifferentialPrivacyParameters,
            sensitivity: float = 1,
            pessimistic_estimate: bool = True,
            sampling_prob: float = 1.0,
            adjacency_type: AdjacencyType = AdjacencyType.REMOVE
    ) -> 'GaussianPrivacyLoss':
        """Creates the privacy loss for Gaussian mechanism with desired privacy.

        Uses binary search to find the smallest possible standard deviation of the
        Gaussian noise for which the mechanism is (epsilon, delta)-differentially
        private, with respect to the REMOVE relation.

        Note: Only the REMOVE adjacency type is used in determining the parameter,
          since for all epsilon > 0, the hockey-stick divergence for PLD with
          respect to the REMOVE adjacency type is at least that for PLD with respect
          to ADD adjacency type.

        Args:
          privacy_parameters: the desired privacy guarantee of the mechanism.
          sensitivity: the sensitivity of function f. (i.e. the maximum absolute
            change in f when an input to a single user changes.)
          pessimistic_estimate: a value indicating whether the rounding is done in
            such a way that the resulting epsilon-hockey stick divergence
            computation gives an upper estimate to the real value.
          sampling_prob: sub-sampling probability, a value in (0,1].
          adjacency_type: type of adjacency relation to used for defining the
            privacy loss distribution.

        Returns:
          The privacy loss of the Gaussian mechanism with the given privacy
          guarantee.
        """
        if privacy_parameters.delta == 0:
            raise ValueError('delta=0 is not allowed for the Gaussian mechanism')

        # The initial standard deviation is set to
        # sqrt(2 * ln(1.5/delta)) * sensitivity / epsilon. It is known that, when
        # epsilon is no more than one, the Gaussian mechanism with this standard
        # deviation is (epsilon, delta)-DP. See e.g. Appendix A in Dwork and Roth
        # book, "The Algorithmic Foundations of Differential Privacy".
        search_parameters = common.BinarySearchParameters(
            0,
            math.inf,
            initial_guess=math.sqrt(2 * math.log(1.5 / privacy_parameters.delta)) *
                          sensitivity / privacy_parameters.epsilon)

        def _get_delta_for_standard_deviation(current_standard_deviation):
            return GaussianPrivacyLoss(
                current_standard_deviation,
                sensitivity=sensitivity,
                sampling_prob=sampling_prob,
                adjacency_type=AdjacencyType.REMOVE).get_delta_for_epsilon(
                privacy_parameters.epsilon)

        standard_deviation = common.inverse_monotone_function(
            _get_delta_for_standard_deviation, privacy_parameters.delta,
            search_parameters)

        return GaussianPrivacyLoss(
            standard_deviation,
            sensitivity=sensitivity,
            pessimistic_estimate=pessimistic_estimate,
            sampling_prob=sampling_prob,
            adjacency_type=adjacency_type)

    @property
    def standard_deviation(self) -> float:
        """The standard deviation of the corresponding Gaussian noise."""
        return self._standard_deviation


class DiscreteLaplacePrivacyLoss(AdditiveNoisePrivacyLoss):
    """Privacy loss of the discrete Laplace mechanism.

    The discrete Laplace mechanism for computing an integer-valued function f
    simply outputs the sum of the true value of the function and a noise drawn
    from the discrete Laplace distribution. Recall that the discrete Laplace
    distribution with parameter a > 0 has probability mass function
    Z * exp(-a * |x|) at x for any integer x, where Z = (e^a - 1) / (e^a + 1).

    This class represents the privacy loss for the aforementioned
    discrete Laplace mechanism with a given parameter, and a given sensitivity of
    the function f. It is assumed that the function f only outputs an integer.
    The privacy loss distribution of the discrete Laplace mechanism is equivalent
    to that between the discrete Laplace distribution and the same distribution
    but shifted by the sensitivity. Specifically, the privacy loss
    distribution of the discrete Laplace mechanism is generated as follows:
    - Let mu = DLap(0, a) be the discrete Laplace noise PMF as given above.
    - Let mu_lower(x) := mu(x - sensitivity), i.e., right shifted by sensitivity
    - Sample x ~ mu_upper = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)), which is equal to
      parameter * (|x - sensitivity| - |x|).

    Case of sub-sampling (Refer to supplementary material for more details):
    The discrete Laplace mechanism with sub-sampling for computing a scalar
    integer-valued function f, first samples a subset of data points including
    each data point independently with probability q, and returns the sum of the
    true values and a noise drawn from the discrete Laplace distribution. Here, we
    consider differential privacy with respect to the addition/removal relation.

    When the sub-sampling probability is q, the worst-case privacy loss
    distribution is generated as follows:
    For ADD adjacency type:
    - Let mu_lower(x) := q * mu(x - sensitivity) + (1-q) * mu(x)
    - Sample x ~ mu_upper = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).
    For REMOVE adjacency type:
    - Let mu_upper(x) := q * mu(x + sensitivity) + (1-q) * mu(x)
    - Sample x ~ mu_lower = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).

    Note: When q = 1, the result privacy loss distributions for both ADD and
      REMOVE adjacency types are identical.
    """

    def __init__(self,
                 parameter: float,
                 sensitivity: int = 1,
                 sampling_prob: float = 1.0,
                 adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> None:
        """Initializes the privacy loss of the discrete Laplace mechanism.

        Args:
          parameter: the parameter of the discrete Laplace distribution.
          sensitivity: the sensitivity of function f. (i.e. the maximum absolute
            change in f when an input to a single user changes.)
          sampling_prob: sub-sampling probability, a value in (0,1].
          adjacency_type: type of adjacency relation to used for defining the
            privacy loss distribution.
        """
        if parameter <= 0:
            raise ValueError(f'Parameter is not a positive real number: {parameter}')

        if not isinstance(sensitivity, int):
            raise ValueError(f'Sensitivity is not an integer : {sensitivity}')

        self._parameter = parameter
        self._discrete_laplace_random_variable = stats.dlaplace(parameter)
        super().__init__(sensitivity, True, sampling_prob, adjacency_type)

    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        """Computes privacy loss at the tail of the discrete Laplace distribution.

        For ADD adjacency type:
        lower_x_truncation = 1 and upper_x_truncation = sensitivity-1

        For REMOVE adjacency type:
        lower_x_truncation = -sensitivity+1 and upper_x_truncation = -1

        The probability mass below lower_x_truncation and above upper_x_truncation
        are computed using mu_upper_cdf.

        Returns:
          A TailPrivacyLossDistribution instance representing the tail of the
          privacy loss distribution.
        """
        if self.adjacency_type == AdjacencyType.ADD:
            lower_x_truncation, upper_x_truncation = 1, self.sensitivity - 1
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            lower_x_truncation, upper_x_truncation = 1 - self.sensitivity, -1
        return TailPrivacyLossDistribution(
            lower_x_truncation, upper_x_truncation, {
                self.privacy_loss(lower_x_truncation - 1):
                    self.mu_upper_cdf(lower_x_truncation - 1),
                self.privacy_loss(upper_x_truncation + 1):
                    1 - self.mu_upper_cdf(upper_x_truncation)
            })

    def privacy_loss_without_subsampling(self, x: float) -> float:
        """Computes privacy loss of the discrete Laplace mechanism without sub-sampling at a given point.

        Args:
          x: the point at which the privacy loss is computed.

        Returns:
          The privacy loss of the discrete Laplace mechanism without sub-sampling at
          integer value x, which is given as
          For ADD adjacency type:    parameter * (|x - sensitivity| - |x|).
          For REMOVE adjacency type: parameter * (|x| - |x + sensitivity|).
        """
        if not isinstance(x, int):
            raise ValueError(f'Privacy loss at x is undefined for x = {x}')

        if self.adjacency_type == AdjacencyType.ADD:
            return (abs(x - self.sensitivity) - abs(x)) * self._parameter
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return (abs(x) - abs(x + self.sensitivity)) * self._parameter

    def inverse_privacy_loss_without_subsampling(self,
                                                 privacy_loss: float) -> float:
        """Computes the inverse of a given privacy loss for the discrete Laplace mechanism.

        Args:
          privacy_loss: the privacy loss value.

        Returns:
          The largest float x such that the privacy loss at x is at least
          privacy_loss.
          For ADD adjacency type:
            If privacy_loss <= - sensitivity * parameter, x is equal to infinity.
            If - sensitivity * parameter < privacy_loss <= sensitivity * parameter,
              x is equal to floor(0.5 * (sensitivity - privacy_loss / parameter)).
            If privacy_loss > sensitivity * parameter, no such x exists and the
              function returns -infinity.
          For REMOVE adjacency type:
            For any value of privacy_loss, x is equal to the corresponding value for
              ADD adjacency type decreased by sensitivity.
        """
        loss_threshold = privacy_loss / self._parameter
        if loss_threshold > self.sensitivity:
            return -math.inf
        if loss_threshold <= -self.sensitivity:
            return math.inf
        if self.adjacency_type == AdjacencyType.ADD:
            return math.floor(0.5 * (self.sensitivity - loss_threshold))
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return math.floor(0.5 * (-self.sensitivity - loss_threshold))

    def noise_cdf(self, x: Union[float,
                                 Iterable[float]]) -> Union[float, np.ndarray]:
        """Computes cumulative density function of the discrete Laplace distribution.

        Args:
          x: the point or points at which the cumulative density function is to be
            calculated.

        Returns:
          The cumulative density function of the discrete Laplace noise at x, i.e.,
          the probability that the discrete Laplace noise is less than or equal to
          x.
        """
        return self._discrete_laplace_random_variable.cdf(x)

    @classmethod
    def from_privacy_guarantee(
            cls,
            privacy_parameters: common.DifferentialPrivacyParameters,
            sensitivity: int = 1,
            sampling_prob: float = 1.0,
            adjacency_type: AdjacencyType = AdjacencyType.REMOVE
    ) -> 'DiscreteLaplacePrivacyLoss':
        """Creates privacy loss for discrete Laplace mechanism with desired privacy.

        Without sub-sampling, the parameter of the Laplace mechanism is simply
          epsilon / sensitivity.
        With sub-sampling probability of q, the parameter is given as below.
          log(1 + (exp(epsilon) - 1)/q) / sensitivity,
        Note: Only the REMOVE adjacency type is used in determining the parameter,
          since for all epsilon > 0, the hockey-stick divergence for PLD with
          respect to the REMOVE adjacency type is at least that for PLD with respect
          to ADD adjacency type.

        Args:
          privacy_parameters: the desired privacy guarantee of the mechanism.
          sensitivity: the sensitivity of function f. (i.e. the maximum absolute
            change in f when an input to a single user changes.)
          sampling_prob: sub-sampling probability, a value in (0,1].
          adjacency_type: type of adjacency relation to used for defining the
            privacy loss distribution.

        Returns:
          The privacy loss of the discrete Laplace mechanism with the given privacy
          guarantee.
        """
        if not isinstance(sensitivity, int):
            raise ValueError(f'Sensitivity is not an integer : {sensitivity}')
        if sensitivity <= 0:
            raise ValueError(
                f'Sensitivity is not a positive real number: {sensitivity}')
        if sampling_prob <= 0 or math.isclose(sampling_prob, 0):
            raise ValueError(
                f'Sampling probability ({sampling_prob}) is equal or too close to 0.')
        parameter = (
                np.log(1 + (np.exp(privacy_parameters.epsilon) - 1) / sampling_prob) /
                sensitivity)

        return DiscreteLaplacePrivacyLoss(
            parameter,
            sensitivity=sensitivity,
            sampling_prob=sampling_prob,
            adjacency_type=adjacency_type)

    @property
    def parameter(self) -> float:
        """The parameter of the corresponding Discrete Laplace noise."""
        return self._parameter


class DiscreteGaussianPrivacyLoss(AdditiveNoisePrivacyLoss):
    """Privacy loss of the discrete Gaussian mechanism.

    The discrete Gaussian mechanism for computing a scalar-valued function f
    simply outputs the sum of the true value of the function and a noise drawn
    from the discrete Gaussian distribution. Recall that the (centered) discrete
    Gaussian distribution with parameter sigma has probability mass function
    proportional to exp(-0.5 x^2/sigma^2) at x for any integer x. Since its
    normalization factor and cumulative density function do not have a closed
    form, we will instead consider the truncated version where the noise x is
    restricted to only be in [-truncated_bound, truncated_bound].

    The privacy loss distribution of the discrete Gaussian mechanism is equivalent
    to the privacy loss distribution between the discrete Gaussian distribution
    and the same distribution but shifted by the sensitivity of f. Specifically,
    the privacy loss distribution of the discrete Gaussian mechanism is generated
    as follows:
    - Let mu = N_Z(0, sigma^2, truncation_bound) be the discrete Gaussian noise
      PMF as given above.
    - Let mu_lower(x) := mu(x - sensitivity), i.e., right shifted by sensitivity
    - Sample x ~ mu_upper = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).
    Note that since we consider the truncated version of the noise, we set the
    privacy loss to infinity when x < -truncation_bound + sensitivity.

    Case of sub-sampling (Refer to supplementary material for more details):
    The discrete Gaussian mechanism with sub-sampling for computing a scalar
    integer-valued function f, first samples a subset of data points including
    each data point independently with probability q, and returns the sum of the
    true values and a noise drawn from the discrete Gaussian distribution. Here,
    we consider differential privacy with respect to the
    addition/removal relation.

    When the sub-sampling probability is q, the worst-case privacy loss
    distribution is generated as follows:
    For ADD adjacency type:
    - Let mu_lower(x) := q * mu(x - sensitivity) + (1-q) * mu(x)
    - Sample x ~ mu_upper = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).
    For REMOVE adjacency type:
    - Let mu_upper(x) := q * mu(x + sensitivity) + (1-q) * mu(x)
    - Sample x ~ mu_lower = mu and let the privacy loss be
      ln(mu_upper(x) / mu_lower(x)).

    Note: When q = 1, the result privacy loss distributions for both ADD and
      REMOVE adjacency types are identical.

    Reference:
    Canonne, Kamath, Steinke. "The Discrete Gaussian for Differential Privacy".
    In NeurIPS 2020.
    """

    def __init__(self,
                 sigma: float,
                 sensitivity: int = 1,
                 truncation_bound: Optional[int] = None,
                 sampling_prob: float = 1.0,
                 adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> None:
        """Initializes the privacy loss of the discrete Gaussian mechanism.

        Args:
          sigma: the parameter of the discrete Gaussian distribution. Note that
            unlike the (continuous) Gaussian distribution this is not equal to the
            standard deviation of the noise.
          sensitivity: the sensitivity of function f. (i.e. the maximum absolute
            change in f when an input to a single user changes.)
          truncation_bound: bound for truncating the noise, i.e. the noise will only
            have a support in [-truncation_bound, truncation_bound]. When not
            specified, truncation_bound will be chosen in such a way that the mass
            of the noise outside of this range is at most 1e-30.
          sampling_prob: sub-sampling probability, a value in (0,1].
          adjacency_type: type of adjacency relation to used for defining the
            privacy loss distribution.
        """
        if sigma <= 0:
            raise ValueError(f'Sigma is not a positive real number: {sigma}')
        if not isinstance(sensitivity, int):
            raise ValueError(f'Sensitivity is not an integer : {sensitivity}')

        self._sigma = sigma
        if truncation_bound is None:
            # Tail bound from Canonne et al. ensures that the mass that gets truncated
            # is at most 1e-30. (See Proposition 1 in the supplementary material.)
            self._truncation_bound = math.ceil(11.6 * sigma)
        else:
            self._truncation_bound = truncation_bound

        if 2 * self._truncation_bound < sensitivity:
            raise ValueError(f'Truncation bound ({truncation_bound}) is smaller '
                             f'than 0.5 * sensitivity (0.5 * {sensitivity})')

        # Create the PMF and CDF.
        self._offset = -1 * self._truncation_bound - 1
        self._pmf_array = np.arange(-1 * self._truncation_bound,
                                    self._truncation_bound + 1)
        self._pmf_array = np.exp(-0.5 * (self._pmf_array) ** 2 / (sigma ** 2))
        self._pmf_array = np.insert(self._pmf_array, 0, 0)
        self._cdf_array = np.add.accumulate(self._pmf_array)
        self._pmf_array /= self._cdf_array[-1]
        self._cdf_array /= self._cdf_array[-1]

        super().__init__(sensitivity, True, sampling_prob, adjacency_type)

    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        """Computes the privacy loss at the tail of the discrete Gaussian distribution.

        The lower_x_truncation and upper_x_truncation are chosen such that for any
        x < lower_x_truncation, the privacy loss is +infinity (or undefined), and
        for any
        x > upper_x_truncation, the privacy loss is -infinity (or undefined).

        With sampling probability of q, the privacy loss tail is given as
        For ADD adjacency type:
        (if q == 1) lower_x_truncation = sensitivity - truncation_bound
        (if q < 1)  lower_x_truncation = - truncation_bound
        In either case, upper_x_truncation = truncation_bound

        For REMOVE adjacency type:
        (if q == 1) upper_x_truncation = truncation_bound - sensitivity
        (if q < 1)  upper_x_truncation = truncation_bound
        In either case, lower_x_truncation = - truncation_bound

        Returns:
          A TailPrivacyLossDistribution instance representing the tail of the
          privacy loss distribution.
        """
        if self.adjacency_type == AdjacencyType.ADD:
            upper_x_truncation = self._truncation_bound
            if self.sampling_prob == 1.0:
                lower_x_truncation = self.sensitivity - self._truncation_bound
            else:
                lower_x_truncation = -1 * self._truncation_bound
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            lower_x_truncation = -1 * self._truncation_bound
            if self.sampling_prob == 1.0:
                upper_x_truncation = self._truncation_bound - self.sensitivity
            else:
                upper_x_truncation = self._truncation_bound

        return TailPrivacyLossDistribution(
            lower_x_truncation, upper_x_truncation,
            {math.inf: self.mu_upper_cdf(lower_x_truncation - 1)})

    def privacy_loss_without_subsampling(self, x: float) -> float:
        """Computes the privacy loss of the discrete Gaussian mechanism without sub-sampling at a given point.

        Args:
          x: the point at which the privacy loss is computed.

        Returns:
          The privacy loss of the discrete Gaussian mechanism at integer value x,
          which is given as

          For ADD adjacency type:
          If x lies in [-truncation_bound + sensitivity, truncation_bound],
            it is equal to sensitivity * (0.5 * sensitivity - x) / sigma^2.
          If x lies in [-truncation_bound, -truncation_bound + sensitivity),
            it is equal to infinity.
          If x lies in (truncation_bound, trunction_bound + sensitivity],
            it is equal to -infinity.
          Otherwise, the privacy loss is undefined (ValueError is raised).

          For REMOVE adjacency type:
           Same as the case of ADD with x replaced by x + sensitivity.

        Raises:
          ValueError: if the privacy loss is undefined.
        """

        def privacy_loss_without_subsampling_for_add(x: float) -> float:
            if (not isinstance(x, int) or x < -1 * self._truncation_bound or
                    x > self._truncation_bound + self.sensitivity):
                actual_x = (
                    x if self.adjacency_type == AdjacencyType.ADD else
                    x - self.sensitivity)
                raise ValueError(f'Privacy loss at x is undefined for x = {actual_x}')
            if x > self._truncation_bound:
                return -math.inf
            if x < self.sensitivity - self._truncation_bound:
                return math.inf
            return self.sensitivity * (0.5 * self.sensitivity - x) / (self._sigma ** 2)

        if self.adjacency_type == AdjacencyType.ADD:
            return privacy_loss_without_subsampling_for_add(x)
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return privacy_loss_without_subsampling_for_add(x + self.sensitivity)

    def inverse_privacy_loss_without_subsampling(self,
                                                 privacy_loss: float) -> float:
        """Computes the inverse of a given privacy loss for the discrete Gaussian mechanism without sub-sampling.

        Args:
          privacy_loss: the privacy loss value.

        Returns:
          The largest int x such that the privacy loss at x is at least
          privacy_loss, which is given as
          For ADD adjacency type:
            floor(0.5 * sensitivity - privacy_loss * sigma^2 / sensitivity) clipped
            to the interval [sensitivity - truncation_bound - 1, truncation_bound].
          For REMOVE adjacency type:
            Same as that for ADD decreased by sensitivity.
        """

        def inverse_privacy_loss_without_subsampling_for_add(
                privacy_loss: float) -> float:
            if privacy_loss == -math.inf:
                return self._truncation_bound
            return math.floor(
                np.clip(
                    0.5 * self.sensitivity - privacy_loss * (self._sigma ** 2) /
                    self.sensitivity,
                    self.sensitivity - self._truncation_bound - 1,
                    self._truncation_bound))

        if self.adjacency_type == AdjacencyType.ADD:
            return inverse_privacy_loss_without_subsampling_for_add(privacy_loss)
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            return (inverse_privacy_loss_without_subsampling_for_add(privacy_loss) -
                    self.sensitivity)

    def noise_cdf(self, x: Union[float,
                                 Iterable[float]]) -> Union[float, np.ndarray]:
        """Computes the cumulative density function of the discrete Gaussian distribution.

        Args:
          x: the point or points at which the cumulative density function is to be
            calculated.

        Returns:
          The cumulative density function of the discrete Gaussian noise at x, i.e.,
          the probability that the discrete Gaussian noise is less than or equal to
          x.
        """
        clipped_x = np.clip(x, -1 * self._truncation_bound - 1,
                            self._truncation_bound)
        indices = np.floor(clipped_x).astype('int') - self._offset
        return self._cdf_array[indices]

    @classmethod
    def from_privacy_guarantee(
            cls,
            privacy_parameters: common.DifferentialPrivacyParameters,
            sensitivity: int = 1,
            sampling_prob: float = 1.0,
            adjacency_type: AdjacencyType = AdjacencyType.REMOVE
    ) -> 'DiscreteGaussianPrivacyLoss':
        """Creates the privacy loss for discrete Gaussian mechanism with desired privacy.

        Uses binary search to find the smallest possible standard deviation of the
        discrete Gaussian noise for which the protocol is (epsilon, delta)-DP.

        Note: Only the REMOVE adjacency type is used in determining the parameter,
          since for all epsilon > 0, the hockey-stick divergence for PLD with
          respect to the REMOVE adjacency type is at least that for PLD with respect
          to ADD adjacency type.

        Args:
          privacy_parameters: the desired privacy guarantee of the mechanism.
          sensitivity: the sensitivity of function f. (i.e. the maximum absolute
            change in f when an input to a single user changes.)
          sampling_prob: sub-sampling probability, a value in (0,1].
          adjacency_type: type of adjacency relation to used for defining the
            privacy loss distribution.

        Returns:
          The privacy loss of the discrete Gaussian mechanism with the given privacy
          guarantee.
        """
        if not isinstance(sensitivity, int):
            raise ValueError(f'Sensitivity is not an integer : {sensitivity}')
        if privacy_parameters.delta == 0:
            raise ValueError('delta=0 is not allowed for discrete Gaussian mechanism')

        # The initial standard deviation is set to
        # sqrt(2 * ln(1.5/delta)) * sensitivity / epsilon. It is known that, when
        # epsilon is no more than one, the (continuous) Gaussian mechanism with this
        # standard deviation is (epsilon, delta)-DP. See e.g. Appendix A in Dwork
        # and Roth book, "The Algorithmic Foundations of Differential Privacy".
        search_parameters = common.BinarySearchParameters(
            0,
            math.inf,
            initial_guess=math.sqrt(2 * math.log(1.5 / privacy_parameters.delta)) *
                          sensitivity / privacy_parameters.epsilon)

        def _get_delta_for_sigma(current_sigma):
            return DiscreteGaussianPrivacyLoss(
                current_sigma,
                sensitivity=sensitivity,
                sampling_prob=sampling_prob,
                adjacency_type=AdjacencyType.REMOVE).get_delta_for_epsilon(
                privacy_parameters.epsilon)

        sigma = common.inverse_monotone_function(_get_delta_for_sigma,
                                                 privacy_parameters.delta,
                                                 search_parameters)

        return DiscreteGaussianPrivacyLoss(
            sigma,
            sensitivity=sensitivity,
            sampling_prob=sampling_prob,
            adjacency_type=adjacency_type)

    def standard_deviation(self) -> float:
        """The standard deviation of the corresponding discrete Gaussian noise."""
        return math.sqrt(
            sum(((i + self._offset) ** 2) * probability_mass
                for i, probability_mass in enumerate(self._pmf_array)))
