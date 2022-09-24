"""Helper functions for privacy accounting across queries."""

import math
import typing
from scipy import special

from fedml.core.dp.budget_accountant import common
from fedml.core.dp.budget_accountant import privacy_loss_distribution
from fedml.core.dp.budget_accountant import privacy_loss_mechanism


def get_smallest_parameter(
        privacy_parameters: common.DifferentialPrivacyParameters, num_queries: int,
        privacy_loss_distribution_constructor: typing.Callable[
            [float], privacy_loss_distribution.PrivacyLossDistribution],
        search_parameters: common.BinarySearchParameters
) -> typing.Union[float, None]:
    """Finds smallest parameter for which the mechanism satisfies desired privacy.

    This function computes the smallest "parameter" for which the corresponding
    mechanism, when run a specified number of times, satisfies a given privacy
    level. It is assumed that, when the parameter increases, the mechanism becomes
    more private.

    Args:
      privacy_parameters: The desired privacy guarantee.
      num_queries: Number of times the mechanism will be invoked.
      privacy_loss_distribution_constructor: A function that takes in a parameter
        and returns the privacy loss distribution for the corresponding mechanism
        for the given parameter.
      search_parameters: Parameters used for binary search.

    Returns:
      Smallest parameter for which the corresponding mechanism with that
      parameter, when applied the given number of times, satisfies the desired
      privacy guarantee. When no parameter in the given range satisfies this,
      return None.
    """

    def get_delta_for_parameter(parameter):
        pld_single_query = privacy_loss_distribution_constructor(parameter)
        pld_all_queries = pld_single_query.self_compose(num_queries)
        return pld_all_queries.get_delta_for_epsilon(privacy_parameters.epsilon)

    return common.inverse_monotone_function(get_delta_for_parameter,
                                            privacy_parameters.delta,
                                            search_parameters)


def get_smallest_laplace_noise(
        privacy_parameters: common.DifferentialPrivacyParameters,
        num_queries: int,
        sensitivity: float = 1) -> float:
    """Finds smallest Laplace noise for which the mechanism satisfies desired privacy.

    Args:
      privacy_parameters: The desired privacy guarantee.
      num_queries: Number of times the mechanism will be invoked.
      sensitivity: The l1 sensitivity of each query.

    Returns:
      Smallest parameter for which the Laplace mechanism with this parameter, when
      applied the given number of times, satisfies the desired privacy guarantee.
    """

    def privacy_loss_distribution_constructor(parameter):
        # Setting value_discretization_interval equal to
        # 0.01 * epsilon / num_queries ensures that the resulting parameter is not
        # (epsilon', delta)-DP for epsilon' less than  0.99 * epsilon / num_queries.
        # This is a heuristic for getting a reasonable pessimistic estimate for the
        # noise parameter.
        return privacy_loss_distribution.from_laplace_mechanism(
            parameter,
            sensitivity=sensitivity,
            value_discretization_interval=0.01 * privacy_parameters.epsilon /
                                          num_queries)

    # Laplace mechanism with parameter sensitivity * num_queries / epsilon is
    # epsilon-DP (for num_queries queries).
    search_parameters = common.BinarySearchParameters(
        0, num_queries * sensitivity / privacy_parameters.epsilon)

    parameter = get_smallest_parameter(privacy_parameters, num_queries,
                                       privacy_loss_distribution_constructor,
                                       search_parameters)
    if parameter is None:
        parameter = num_queries * sensitivity / privacy_parameters.epsilon
    return parameter


def get_smallest_discrete_laplace_noise(
        privacy_parameters: common.DifferentialPrivacyParameters,
        num_queries: int,
        sensitivity: int = 1) -> float:
    """Finds smallest discrete Laplace noise for which the mechanism satisfies desired privacy.

    Note that from the way discrete Laplace distribution is defined, the amount of
    noise decreases as the parameter increases. (In other words, the mechanism
    becomes less private as the parameter increases.) As a result, the output will
    be the largest parameter (instead of smallest as in Laplace).

    Args:
      privacy_parameters: The desired privacy guarantee.
      num_queries: Number of times the mechanism will be invoked.
      sensitivity: The l1 sensitivity of each query.

    Returns:
      Largest parameter for which the discrete Laplace mechanism with this
      parameter, when applied the given number of times, satisfies the desired
      privacy guarantee.
    """

    # Search for inverse of the parameter instead of the parameter itself.
    def privacy_loss_distribution_constructor(inverse_parameter):
        parameter = 1 / inverse_parameter
        # Setting value_discretization_interval equal to parameter because the
        # privacy loss of discrete Laplace mechanism is always divisible by the
        # parameter.
        return privacy_loss_distribution.from_discrete_laplace_mechanism(
            parameter,
            sensitivity=sensitivity,
            value_discretization_interval=parameter)

    # discrete Laplace mechanism with parameter
    # epsilon / (sensitivity * num_queries) is epsilon-DP (for num_queries
    # queries).
    search_parameters = common.BinarySearchParameters(
        0, num_queries * sensitivity / privacy_parameters.epsilon)

    inverse_parameter = get_smallest_parameter(
        privacy_parameters, num_queries, privacy_loss_distribution_constructor,
        search_parameters)
    if inverse_parameter is None:
        parameter = privacy_parameters.epsilon / (num_queries * sensitivity)
    else:
        parameter = 1 / inverse_parameter
    return parameter


def get_smallest_gaussian_noise(
        privacy_parameters: common.DifferentialPrivacyParameters,
        num_queries: int,
        sensitivity: float = 1) -> float:
    """Finds smallest Gaussian noise for which the mechanism satisfies desired privacy.

    Args:
      privacy_parameters: The desired privacy guarantee.
      num_queries: Number of times the mechanism will be invoked.
      sensitivity: The l2 sensitivity of each query.

    Returns:
      Smallest standard deviation for which the Gaussian mechanism with this std,
      when applied the given number of times, satisfies the desired privacy
      guarantee.
    """
    # The l2 sensitivity grows as square root of the number of queries
    return privacy_loss_mechanism.GaussianPrivacyLoss.from_privacy_guarantee(
        privacy_parameters,
        sensitivity=sensitivity * math.sqrt(num_queries)).standard_deviation


def advanced_composition(
        privacy_parameters: common.DifferentialPrivacyParameters,
        num_queries: int, total_delta: float) -> typing.Optional[float]:
    """Computes total DP parameters after applying an algorithm with given privacy parameters multiple times.

    Using the optimal advanced composition theorem, Theorem 3.3 from the paper
    Kairouz, Oh, Viswanath. "The Composition Theorem for Differential Privacy",
    to compute the total DP parameters given that we are applying an algorithm
    with a given privacy parameters for a given number of times.

    Note that we can compute this alternatively from PrivacyLossDistribution
    by invoking from_privacy_parameters and applying the given number of
    composition. When setting value_discretization_interval appropriately, these
    two approaches should coincide but using the advanced composition theorem
    directly is less computational intensive.

    Args:
      privacy_parameters: The privacy guarantee of a single query.
      num_queries: Number of times the algorithm is invoked.
      total_delta: The target value of total delta of the privacy parameters for
        the multiple runs of the algorithm.

    Returns:
      total_epsilon such that, when applying the algorithm the given number of
      times, the result is still (total_epsilon, total_delta)-DP.

      None when the total_delta is less than 1 - (1 - delta)^num_queries, for
      which no guarantee of (total_epsilon, total_delta)-DP is possible for any
      value of total_epsilon.

    ..
        The calculation follows Theorem 3.3 of
          [KOV17] Kairouz, Peter, Sewoong Oh, and Pramod Viswanath. "The composition theorem for differential privacy."
        IEEE Transactions on Information Theory 63.6 (2017): 4037-4049.https://arxiv.org/pdf/1311.0776.pdf
    """
    epsilon = privacy_parameters.epsilon
    delta = privacy_parameters.delta
    k = num_queries

    # The calculation follows Theorem 3.3 of https://arxiv.org/pdf/1311.0776.pdf
    for i in range(k // 2, -1, -1):
        delta_i = 0
        for l in range(i):
            delta_i += special.binom(k, l) * (
                    math.exp(epsilon * (k - l)) - math.exp(epsilon * (k - 2 * i + l)))
        delta_i /= ((1 + math.exp(epsilon)) ** k)
        if 1 - ((1 - delta) ** k) * (1 - delta_i) <= total_delta:
            return epsilon * (k - 2 * i)
    return None


def get_smallest_epsilon_from_advanced_composition(
        total_privacy_parameters: common.DifferentialPrivacyParameters,
        num_queries: int, delta: float = 0) -> typing.Optional[float]:
    """Computes DP parameters that after a certain number of queries remain DP with given parameters.

    Using the optimal advanced composition theorem, Theorem 3.3 from the paper
    Kairouz, Oh, Viswanath. "The Composition Theorem for Differential Privacy",
    to compute DP parameter for an algorithm, so that when applied a given number
    of times it remains DP with given privacy parameters.

    Args:
      total_privacy_parameters: The desired privacy guarantee after applying the
        algorithm a given number of times.
      num_queries: Number of times the algorithm is invoked.
      delta: The value of DP parameter delta for the algorithm.

    Returns:
      epsilon such that if an algorithm is (epsilon, delta)-DP, then applying it
      the given number of times remains DP with total_privacy_parameters.

      None when total_privacy_parameters.delta is less than
      1 - (1 - delta)^num_queries for which no guarantee of
      total_privacy_parameters DP is possible for any value of epsilon.
    """
    if 1 - ((1 - delta) ** num_queries) > total_privacy_parameters.delta:
        return None

    search_parameters = common.BinarySearchParameters(
        total_privacy_parameters.epsilon / num_queries,
        total_privacy_parameters.epsilon)

    def get_total_epsilon_for_epsilon(epsilon):
        privacy_parameters = common.DifferentialPrivacyParameters(epsilon, delta)
        return advanced_composition(privacy_parameters, num_queries,
                                    total_privacy_parameters.delta)

    return common.inverse_monotone_function(
        get_total_epsilon_for_epsilon,
        total_privacy_parameters.epsilon,
        search_parameters,
        increasing=True)
