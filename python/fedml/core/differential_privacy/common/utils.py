from numbers import Real


def check_bounds(lower, upper):
    if not isinstance(lower, Real) or not isinstance(upper, Real):
        raise TypeError("Bounds must be numeric")
    if lower > upper:
        raise ValueError("Lower bound must not be greater than upper bound")
    return lower, upper


def check_numeric_value(value):
    if not isinstance(value, Real):
        raise TypeError("Value to be randomised must be a number")
    return True


def check_params(epsilon, delta, sensitivity):
    if not isinstance(epsilon, Real) or not isinstance(delta, Real):
        raise TypeError("Epsilon and delta must be numeric")
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative")
    if delta < 0 or delta > 1:
        raise ValueError("Delta must be in [0, 1]")
    if epsilon + delta == 0:
        raise ValueError("Epsilon and Delta cannot both be zero")
    if not isinstance(sensitivity, Real):
        raise TypeError("Sensitivity must be numeric")
    if sensitivity < 0:
        raise ValueError("Sensitivity must be non-negative")
