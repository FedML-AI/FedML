import secrets
from numbers import Real, Integral

"""
Common functions for DP. Some codes refer to diffprivlib: https://github.com/IBM/differential-privacy-library
"""


def check_bounds(lower, upper):
    """
    Check if the provided lower and upper bounds are valid.

    Args:
        lower (Real): The lower bound.
        upper (Real): The upper bound.

    Returns:
        Tuple[Real, Real]: A tuple containing the validated lower and upper bounds.

    Raises:
        TypeError: If lower or upper is not a numeric type.
        ValueError: If the lower bound is greater than the upper bound.
    """
    if not isinstance(lower, Real) or not isinstance(upper, Real):
        raise TypeError("Bounds must be numeric")
    if lower > upper:
        raise ValueError("Lower bound must not be greater than upper bound")
    return lower, upper


def check_numeric_value(value):
    """
    Check if the provided value is a numeric type.

    Args:
        value (Real): The value to be checked.

    Returns:
        bool: True if the value is numeric, False otherwise.

    Raises:
        TypeError: If the value is not a numeric type.
    """
    if not isinstance(value, Real):
        raise TypeError("Value to be randomised must be a number")
    return True


def check_integer_value(value):
    """
    Check if the provided value is an integer.

    Args:
        value (Integral): The value to be checked.

    Returns:
        bool: True if the value is an integer, False otherwise.

    Raises:
        TypeError: If the value is not an integer.
    """
    if not isinstance(value, Integral):
        raise TypeError("Value to be randomised must be an integer")
    return True


def check_epsilon_delta(epsilon, delta, allow_zero=False):
    """
    Check if the provided epsilon and delta values are valid for differential privacy.

    Args:
        epsilon (Real): Epsilon value.
        delta (Real): Delta value.
        allow_zero (bool, optional): Whether to allow epsilon and delta to be zero. Default is False.

    Raises:
        TypeError: If epsilon or delta is not a numeric type.
        ValueError: If epsilon is negative, delta is outside [0, 1] range, or both epsilon and delta are zero.
    """
    if not isinstance(epsilon, Real) or not isinstance(delta, Real):
        raise TypeError("Epsilon and delta must be numeric")
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative")
    if delta < 0 or float(delta) > 1.0:
        raise ValueError("Delta must be in [0, 1]")
    if not allow_zero and epsilon + delta == 0:
        raise ValueError("Epsilon and Delta cannot both be zero")


def check_params(epsilon, delta, sensitivity):
    """
    Check the validity of epsilon, delta, and sensitivity parameters for differential privacy.

    Args:
        epsilon (Real): Epsilon value.
        delta (Real): Delta value.
        sensitivity (Real): Sensitivity value.

    Raises:
        TypeError: If epsilon, delta, or sensitivity is not a numeric type.
        ValueError: If epsilon is negative, delta is outside [0, 1] range, or sensitivity is negative.
    """
    check_epsilon_delta(epsilon, delta, allow_zero=False)
    if not isinstance(sensitivity, Real):
        raise TypeError("Sensitivity must be numeric")
    if sensitivity < 0:
        raise ValueError("Sensitivity must be non-negative")


def bernoulli_neg_exp(gamma, rng=None):
    """Sample from Bernoulli(exp(-gamma)).
    "The Discrete Gaussian for Differential Privacy": https://arxiv.org/pdf/2004.00010v2.pdf

    Parameters
    ----------
    gamma : float. Parameter to sample from Bernoulli(exp(-gamma)).  Must be non-negative.
    rng : Random number generator, optional. Random number generator to use.
            If not provided, uses SystemRandom from secrets by default.
    Returns
    -------
    One sample from the Bernoulli(exp(-gamma)) distribution.
    """
    if gamma < 0:
        raise ValueError(f"Gamma must be non-negative, got {gamma}.")
    if rng is None:
        rng = secrets.SystemRandom()
    while gamma > 1:
        gamma -= 1
        if not bernoulli_neg_exp(1, rng):
            return 0
    counter = 1
    while rng.random() <= gamma / counter:
        counter += 1
    return counter % 2
