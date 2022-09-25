"""Probability mass function for privacy loss distributions.

This file implements work the privacy loss distribution (PLD) probability mass
functions (PMF)and its basic functionalities. Please refer to the
supplementary material below for more details:
../../common_docs/Privacy_Loss_Distributions.pdf
"""

import abc
import itertools
import math
from typing import Iterable, List, Mapping, Tuple, Union
import numpy as np
from scipy import signal

from fedml.core.dp.budget_accountant import common

ArrayLike = Union[np.ndarray, List[float]]
_MAX_PMF_SPARSE_SIZE = 1000


def _get_delta_for_epsilon(infinity_mass: float,
                           reversed_losses: Iterable[float],
                           probs: Iterable[float], epsilon: float) -> float:
    """Computes the epsilon-hockey stick divergence.

    Args:
      infinity_mass: the probability of the infinite loss.
      reversed_losses: privacy losses, assumed to be sorted in descending order.
      probs: probabilities corresponding to losses.
      epsilon: the epsilon in the epsilon-hockey stick divergence.

    Returns:
      The epsilon-hockey stick divergence.
    """
    delta = 0
    for loss, prob in zip(reversed_losses, probs):
        if loss <= epsilon:
            break
        delta += (1 - np.exp(epsilon - loss)) * prob
    return delta + infinity_mass


def _get_epsilon_for_delta(infinity_mass: float,
                           reversed_losses: Iterable[float],
                           probs: Iterable[float], delta: float) -> float:
    """Computes epsilon for which hockey stick divergence is at most delta.

    Args:
      infinity_mass: the probability of the infinite loss.
      reversed_losses: privacy losses, assumed to be sorted in descending order.
      probs: probabilities corresponding to losses.
      delta: the target epsilon-hockey stick divergence..

    Returns:
       The smallest epsilon such that the epsilon-hockey stick divergence is at
       most delta. When no such finite epsilon exists, return math.inf.
    """
    if infinity_mass > delta:
        return math.inf

    mass_upper, mass_lower = infinity_mass, 0

    for loss, prob in zip(reversed_losses, probs):
        if (mass_upper > delta and mass_lower > 0 and math.log(
                (mass_upper - delta) / mass_lower) >= loss):
            # Epsilon is greater than or equal to loss.
            break

        mass_upper += prob
        mass_lower += math.exp(-loss) * prob

        if mass_upper >= delta and mass_lower == 0:
            # This only occurs when loss is very large, which results in exp(-loss)
            # being treated as zero.
            return max(0, loss)

    if mass_upper <= mass_lower + delta:
        return 0
    return math.log((mass_upper - delta) / mass_lower)


def _truncate_tails(probs: ArrayLike, tail_mass_truncation: float,
                    pessimistic_estimate: bool) -> Tuple[int, ArrayLike, float]:
    """Truncates an array from both sides by not more than tail_mass_truncation.

    It truncates the maximum prefix and suffix from probs, each of which have
    sum <= tail_mass_truncation/2.

    Args:
      probs: array to truncate.
      tail_mass_truncation: an upper bound on the tails of the probability mass of
        the PMF that might be truncated.
      pessimistic_estimate: if true then the left truncated sum is added to 0th
        element of the truncated array and the right truncated returned as it goes
        to infinity. If false then the right truncated sum is added to the last of
        the truncated array and the left truncated sum is discarded.

    Returns:
      Tuple of (size of truncated prefix, truncated array, mass that goes to
      infinity).
    """
    if tail_mass_truncation == 0:
        return 0, probs, 0

    def _find_prefix_to_truncate(arr: np.ndarray, threshold: float) -> int:
        # Find the max size of array prefix, with the sum of elements less than
        # threshold.
        s = 0
        for i, val in enumerate(arr):
            s += val
            if s > threshold:
                return i
        return len(arr)

    left_idx = _find_prefix_to_truncate(probs, tail_mass_truncation / 2)
    right_idx = len(probs) - _find_prefix_to_truncate(
        np.flip(probs), tail_mass_truncation / 2)
    # Be sure that left_idx <= right_idx. left_idx > right_idx might be when
    # tail_mass_truncation is too large or if probs has too small mass
    # (i.e. if a few truncations were operated on it already).
    right_idx = max(right_idx, left_idx)

    left_mass = np.sum(probs[:left_idx])
    right_mass = np.sum(probs[right_idx:])

    truncated_probs = probs[left_idx:right_idx]
    if pessimistic_estimate:
        # put truncated the left mass to the 0th element.
        truncated_probs[0] += left_mass
        return left_idx, truncated_probs, right_mass
    # This is rounding to left case. Put truncated the right mass to the last
    # element.
    truncated_probs[-1] += right_mass
    return left_idx, truncated_probs, 0


class PLDPmf(abc.ABC):
    """Base class for probability mass functions for privacy loss distributions.

    The privacy loss distribution (PLD) of two discrete distributions, the upper
    distribution mu_upper and the lower distribution mu_lower, is defined as a
    distribution on real numbers generated by first sampling an outcome o
    according to mu_upper and then outputting the privacy loss
    ln(mu_upper(o) / mu_lower(o)) where mu_lower(o) and mu_upper(o) are the
    probability masses of o in mu_lower and mu_upper respectively. This class
    allows one to create and manipulate privacy loss distributions.

    PLD allows one to (approximately) compute the epsilon-hockey stick divergence
    between mu_upper and mu_lower, which is defined as
    sum_{o} [mu_upper(o) - e^{epsilon} * mu_lower(o)]_+. This quantity in turn
    governs the parameter delta of (eps, delta)-differential privacy of the
    corresponding protocol. (See Observation 1 in the supplementary material.)

    The above definitions extend to continuous distributions. The PLD of two
    continuous distributions mu_upper and mu_lower is defined as a distribution on
    real numbers generated by first sampling an outcome o according to mu_upper
    and then outputting the privacy loss ln(f_{mu_upper}(o) / f_{mu_lower}(o))
    where f_{mu_lower}(o) and f_{mu_upper}(o) are the probability density
    functions at o in mu_lower and mu_upper respectively. Moreover, for continuous
    distributions the epsilon-hockey stick divergence is defined as
    integral [f_{mu_upper}(o) - e^{epsilon} * f_{mu_lower}(o)]_+ do.
    """

    def __init__(self, discretization: float, infinity_mass: float,
                 pessimistic_estimate: bool):
        self._discretization = discretization
        self._infinity_mass = infinity_mass
        self._pessimistic_estimate = pessimistic_estimate

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Returns number of points in discretization."""

    @abc.abstractmethod
    def compose(self,
                other: 'PLDPmf',
                tail_mass_truncation: float = 0) -> 'PLDPmf':
        """Computes a PMF resulting from composing two PMFs.

        Args:
          other: the privacy loss distribution PMF to be composed. The two must have
            the same discretization and pessimistic_estimate.
          tail_mass_truncation: an upper bound on the tails of the probability mass
            of the PMF that might be truncated.

        Returns:
          A PMF which is the result of convolving (composing) the two.
        """

    @abc.abstractmethod
    def self_compose(self,
                     num_times: int,
                     tail_mass_truncation: float = 0) -> 'PLDPmf':
        """Computes PMF resulting from repeated composing the PMF with itself.

        Args:
          num_times: the number of times to compose this PMF with itself.
          tail_mass_truncation: an upper bound on the tails of the probability mass
            of the PMF that might be truncated.

        Returns:
          A privacy loss distribution PMF which is the result of the composition.
        """

    @abc.abstractmethod
    def get_delta_for_epsilon(self, epsilon: float) -> float:
        """Computes the epsilon-hockey stick divergence."""

    @abc.abstractmethod
    def get_epsilon_for_delta(self, delta: float) -> float:
        """Computes epsilon for which hockey stick divergence is at most delta."""

    @abc.abstractmethod
    def to_dense_pmf(self) -> 'DensePLDPmf':
        """Returns the dense PMF with data from 'self'."""

    @abc.abstractmethod
    def get_delta_for_epsilon_for_composed_pld(self, other: 'PLDPmf',
                                               epsilon: float) -> float:
        """Computes delta for 'epsilon' for the composiion of 'self' and 'other'."""

    def validate_composable(self, other: 'PLDPmf'):
        """Checks whether 'self' and 'other' can be composed."""
        if not isinstance(self, type(other)):
            raise ValueError(f'Only PMFs of the same type can be composed:'
                             f'{type(self).__name__} != {type(other).__name__}.')
        # pylint: disable=protected-access
        if self._discretization != other._discretization:
            raise ValueError(f'Discretization intervals are different: '
                             f'{self._discretization} != '
                             f'{other._discretization}.')
        if self._pessimistic_estimate != other._pessimistic_estimate:
            raise ValueError(f'Estimation types are different: '
                             f'{self._pessimistic_estimate} != '
                             f'{other._pessimistic_estimate}.')  # pylint: disable=protected-access
        # pylint: enable=protected-access


class DensePLDPmf(PLDPmf):
    """Class for dense probability mass function.

    It represents a discrete probability distribution on a grid of privacy losses.
    The grid contains numbers multiple of 'discretization', starting from
    lower_loss * discretization.
    """

    def __init__(self, discretization: float, lower_loss: int, probs: np.ndarray,
                 infinity_mass: float, pessimistic_estimate: bool):
        super().__init__(discretization, infinity_mass, pessimistic_estimate)
        self._lower_loss = lower_loss
        self._probs = probs

    @property
    def size(self) -> int:
        return len(self._probs)

    def compose(self,
                other: 'DensePLDPmf',
                tail_mass_truncation: float = 0) -> 'DensePLDPmf':
        """Computes a PMF resulting from composing two PMFs. See base class."""
        self.validate_composable(other)

        # pylint: disable=protected-access
        lower_loss = self._lower_loss + other._lower_loss
        probs = signal.fftconvolve(self._probs, other._probs)
        infinity_mass = 1 - (1 - self._infinity_mass) * (1 - other._infinity_mass)
        offset, probs, right_tail = _truncate_tails(probs, tail_mass_truncation,
                                                    self._pessimistic_estimate)
        # pylint: enable=protected-access
        return DensePLDPmf(self._discretization, lower_loss + offset, probs,
                           infinity_mass + right_tail, self._pessimistic_estimate)

    def self_compose(self,
                     num_times: int,
                     tail_mass_truncation: float = 1e-15) -> 'DensePLDPmf':
        """See base class."""
        if num_times <= 0:
            raise ValueError(f'num_times should be >= 1, num_times={num_times}')
        lower_loss = self._lower_loss * num_times
        truncation_lower_bound, probs = common.self_convolve(
            self._probs, num_times, tail_mass_truncation)
        lower_loss += truncation_lower_bound
        probs = np.array(probs)
        inf_prob = 1 - (1 - self._infinity_mass) ** num_times
        offset, probs, right_tail = _truncate_tails(probs, tail_mass_truncation,
                                                    self._pessimistic_estimate)
        return DensePLDPmf(self._discretization, lower_loss + offset, probs,
                           inf_prob + right_tail, self._pessimistic_estimate)

    def get_delta_for_epsilon(self, epsilon: float) -> float:
        """Computes the epsilon-hockey stick divergence."""
        upper_loss = (self._lower_loss + len(self._probs) -
                      1) * self._discretization
        reversed_losses = itertools.count(upper_loss, -self._discretization)
        return _get_delta_for_epsilon(self._infinity_mass, reversed_losses,
                                      np.flip(self._probs), epsilon)

    def get_epsilon_for_delta(self, delta: float) -> float:
        """Computes epsilon for which hockey stick divergence is at most delta."""
        upper_loss = (self._lower_loss + len(self._probs) -
                      1) * self._discretization
        reversed_losses = itertools.count(upper_loss, -self._discretization)
        return _get_epsilon_for_delta(self._infinity_mass, reversed_losses,
                                      np.flip(self._probs), delta)

    def to_dense_pmf(self) -> 'DensePLDPmf':
        return self

    def get_delta_for_epsilon_for_composed_pld(self, other: PLDPmf,
                                               epsilon: float) -> float:
        other = other.to_dense_pmf()
        self.validate_composable(other)
        discretization = self._discretization
        # pylint: disable=protected-access
        self_loss = lambda index: (index + self._lower_loss) * discretization
        other_loss = lambda index: (index + other._lower_loss) * discretization

        self_probs, other_probs = self._probs, other._probs
        len_self, len_other = len(self_probs), len(other_probs)
        delta = 1 - (1 - self._infinity_mass) * (1 - other._infinity_mass)
        # pylint: enable=protected-access

        # Compute the hockey stick divergence using equation (2) in the
        # supplementary material. upper_mass represents summation in equation (3)
        # and lower_mass represents the summation in equation (4).

        if self_loss(len_self - 1) + other_loss(len_other - 1) <= epsilon:
            return delta

        i, j = 0, len_other - 1
        upper_mass = lower_mass = 0

        # This is summation by i,j, such that self_loss(i) + other_loss(j) >=
        # epsilon, and self_loss(i) + other_loss(j-1)< epsilon, as in the
        # equation(2).

        # If i is todo small then increase it.
        while self_loss(i) + other_loss(j) < epsilon:
            i += 1

        # Else if j is too large then decrease it.
        while j >= 0 and self_loss(i) + other_loss(j - 1) >= epsilon:
            upper_mass += other_probs[j]
            lower_mass += other_probs[j] * np.exp(-other_loss(j))
            j -= 1

        # Invariant:
        # self_loss(i) + other_loss(j-1) < epsilon <= self_loss(i) + other_loss(j)
        # Sum over all i, keeping this invariant.
        for i in range(i, len_self):
            if j >= 0:
                upper_mass += other_probs[j]
                lower_mass += other_probs[j] * np.exp(-other_loss(j))
            j -= 1
            delta += self_probs[i] * (
                    upper_mass - np.exp(epsilon - self_loss(i)) * lower_mass)

        return delta


class SparsePLDPmf(PLDPmf):
    """Class for sparse probability mass function.

    It represents a discrete probability distribution on a grid of 1d losses with
    a dictionary. The grid contains numbers multiples of 'discretization'.
    """

    def __init__(self, loss_probs: Mapping[int, float], discretization: float,
                 infinity_mass: float, pessimistic_estimate: bool):
        super().__init__(discretization, infinity_mass, pessimistic_estimate)
        self._loss_probs = loss_probs

    @property
    def size(self) -> int:
        return len(self._loss_probs)

    def compose(self,
                other: 'SparsePLDPmf',
                tail_mass_truncation: float = 0) -> 'SparsePLDPmf':
        """Computes a PMF resulting from composing two PMFs. See base class."""
        self.validate_composable(other)
        # Assumed small number of points, so simple quadratic algorithm is fine.
        convolution = {}
        # pylint: disable=protected-access
        for key1, value1 in self._loss_probs.items():
            for key2, value2 in other._loss_probs.items():
                key = key1 + key2
                convolution[key] = convolution.get(key, 0.0) + value1 * value2
        infinity_mass = 1 - (1 - self._infinity_mass) * (1 - other._infinity_mass)
        # pylint: enable=protected-access
        # Do truncation.
        sorted_losses = sorted(convolution.keys())
        probs = [convolution[loss] for loss in sorted_losses]
        offset, probs, right_mass = _truncate_tails(probs, tail_mass_truncation,
                                                    self._pessimistic_estimate)
        sorted_losses = sorted_losses[offset:offset + len(probs)]
        truncated_convolution = dict(zip(sorted_losses, probs))
        return SparsePLDPmf(truncated_convolution, self._discretization,
                            infinity_mass + right_mass, self._pessimistic_estimate)

    def self_compose(self,
                     num_times: int,
                     tail_mass_truncation: float = 1e-15) -> 'PLDPmf':
        """See base class."""
        if num_times <= 0:
            raise ValueError(f'num_times should be >= 1, num_times={num_times}')
        if num_times == 1:
            return self

        # Compute a rough upper bound overestimate, since from some power, the PMF
        # becomes dense and start growing linearly further. But in this case we
        # should definitely go to dense.
        max_result_size = self.size ** num_times

        if max_result_size > _MAX_PMF_SPARSE_SIZE:
            # The size of composed PMF is too large for sparse. Convert to dense.
            return self.to_dense_pmf().self_compose(num_times, tail_mass_truncation)

        result = self
        for i in range(2, num_times + 1):
            # To truncate only on the last composition.
            mass_truncation = 0 if i != num_times else tail_mass_truncation
            result = result.compose(self, mass_truncation)

        return result

    def _get_reversed_losses_probs(self) -> Tuple[List[float], List[float]]:
        """Returns losses, sorted in reverse order and respective probabilities."""
        reversed_losses = sorted(list(self._loss_probs.keys()), reverse=True)
        reversed_probs = [self._loss_probs[loss] for loss in reversed_losses]
        reversed_losses = [loss * self._discretization for loss in reversed_losses]
        return reversed_losses, reversed_probs

    def get_delta_for_epsilon(self, epsilon: float) -> float:
        """Computes the epsilon-hockey stick divergence."""
        reversed_losses, reversed_probs = self._get_reversed_losses_probs()
        return _get_delta_for_epsilon(self._infinity_mass, reversed_losses,
                                      reversed_probs, epsilon)

    def get_epsilon_for_delta(self, delta: float) -> float:
        """Computes epsilon for which hockey stick divergence is at most delta."""
        reversed_losses, reversed_probs = self._get_reversed_losses_probs()
        return _get_epsilon_for_delta(self._infinity_mass, reversed_losses,
                                      reversed_probs, delta)

    def get_delta_for_epsilon_for_composed_pld(self, other: PLDPmf,
                                               epsilon: float) -> float:
        # If 'self' is sparse, then it is small, so it is not so expensive to
        # convert to dense. Let us convert it for simplicity for dense.
        return self.to_dense_pmf().get_delta_for_epsilon_for_composed_pld(
            other, epsilon)

    def to_dense_pmf(self) -> DensePLDPmf:
        """"Converts to dense PMF."""
        lower_loss, probs = common.dictionary_to_list(self._loss_probs)
        return DensePLDPmf(self._discretization, lower_loss, np.array(probs),
                           self._infinity_mass, self._pessimistic_estimate)


def create_pmf(loss_probs: Mapping[int, float], discretization: float,
               infinity_mass: float, pessimistic_estimate: bool) -> PLDPmf:
    """Creates PLDPmfs.

    It returns SparsePLDPmf if the size of loss_probs less than
     MAX_PMF_SPARSE_SIZE, otherwise DensePLDPmf.

    Args:
      loss_probs: probability mass function of the discretized privacy loss
        distribution.
      discretization: the interval length for which the values of the privacy loss
        distribution are discretized.
      infinity_mass: infinity_mass for privacy loss distribution.
      pessimistic_estimate: whether the rounding is done in such a way that the
        resulting epsilon-hockey stick divergence computation gives an upper
        estimate to the real value.

    Returns:
      Created PLDPmf.
    """
    if len(loss_probs) <= _MAX_PMF_SPARSE_SIZE:
        return SparsePLDPmf(loss_probs, discretization, infinity_mass,
                            pessimistic_estimate)

    lower_loss, probs = common.dictionary_to_list(loss_probs)
    probs = np.array(probs)
    return DensePLDPmf(discretization, lower_loss, probs, infinity_mass,
                       pessimistic_estimate)


def compose_pmfs(pmf1: PLDPmf,
                 pmf2: PLDPmf,
                 tail_mass_truncation: float = 0) -> PLDPmf:
    """Computes a PMF resulting from composing two PMFs.

    It returns SparsePLDPmf only if input PLDPmfs are SparsePLDPmf and the
    product of input pmfs sizes are less than MAX_PMF_SPARSE_SIZE.

    Args:
      pmf1: the privacy loss distribution PMF to be composed.
      pmf2: the privacy loss distribution PMF to be composed. The two must have
        the same discretization and pessimistic_estimate.
      tail_mass_truncation: an upper bound on the tails of the probability mass of
        the PMF that might be truncated.

    Returns:
      A PMF which is the result of convolving (composing) the two.
    """
    max_result_size = pmf1.size * pmf2.size
    if (isinstance(pmf1, SparsePLDPmf) and isinstance(pmf2, SparsePLDPmf) and
            max_result_size <= _MAX_PMF_SPARSE_SIZE):
        return pmf1.compose(pmf2, tail_mass_truncation)

    pmf1 = pmf1.to_dense_pmf()
    pmf2 = pmf2.to_dense_pmf()
    return pmf1.compose(pmf2, tail_mass_truncation)
