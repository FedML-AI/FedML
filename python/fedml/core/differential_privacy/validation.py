# MIT License
#
# Copyright (C) IBM Corporation 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Validation functions for the differential privacy library
"""
from numbers import Real, Integral

import numpy as np

from fedml.core.differential_privacy.utils import warn_unused_args


def check_epsilon_delta(epsilon, delta, allow_zero=False):
    """Checks that epsilon and delta are valid values for differential privacy.  Throws an error if checks fail,
    otherwise returns nothing.

    As well as the requirements of epsilon and delta separately, both cannot be simultaneously zero, unless
    ``allow_zero`` is set to ``True``.

    Parameters
    ----------
    epsilon : float
        Epsilon parameter for differential privacy.  Must be non-negative.

    delta : float
        Delta parameter for differential privacy.  Must be on the unit interval, [0, 1].

    allow_zero : bool, default: False
        Allow epsilon and delta both be zero.

    """
    if not isinstance(epsilon, Real) or not isinstance(delta, Real):
        raise TypeError("Epsilon and delta must be numeric")

    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative")

    if not 0 <= delta <= 1:
        raise ValueError("Delta must be in [0, 1]")

    if not allow_zero and epsilon + delta == 0:
        raise ValueError("Epsilon and Delta cannot both be zero")


def check_bounds(bounds, shape=0, min_separation=0.0, dtype=float):
    """Input validation for the ``bounds`` parameter.

    Checks that ``bounds`` is composed of a list of tuples of the form (lower, upper), where lower <= upper and both
    are numeric.  Also checks that ``bounds`` contains the appropriate number of dimensions, and that there is a
    ``min_separation`` between the bounds.

    Parameters
    ----------
    bounds : tuple
        Tuple of bounds of the form (min, max). `min` and `max` can either be scalars or 1-dimensional arrays.

    shape : int, default: 0
        Number of dimensions to be expected in ``bounds``.

    min_separation : float, default: 0.0
        The minimum separation between `lower` and `upper` of each dimension.  This separation is enforced if not
        already satisfied.

    dtype : data-type, default: float
        Data type of the returned bounds.

    Returns
    -------
    bounds : tuple

    """
    if not isinstance(bounds, tuple):
        raise TypeError(f"Bounds must be specified as a tuple of (min, max), got {type(bounds)}.")
    if not isinstance(shape, Integral):
        raise TypeError(f"shape parameter must be integer-valued, got {type(shape)}.")

    lower, upper = bounds

    if np.asarray(lower).size == 1 or np.asarray(upper).size == 1:
        lower = np.ravel(lower).astype(dtype)
        upper = np.ravel(upper).astype(dtype)
    else:
        lower = np.asarray(lower, dtype=dtype)
        upper = np.asarray(upper, dtype=dtype)

    if lower.shape != upper.shape:
        raise ValueError("lower and upper bounds must be the same shape array")
    if lower.ndim > 1:
        raise ValueError("lower and upper bounds must be scalar or a 1-dimensional array")
    if lower.size not in (1, shape):
        raise ValueError(f"lower and upper bounds must have {shape or 1} element(s), got {lower.size}.")

    n_bounds = lower.shape[0]

    for i in range(n_bounds):
        _lower = lower[i]
        _upper = upper[i]

        if not isinstance(_lower, Real) or not isinstance(_upper, Real):
            raise TypeError(f"Each bound must be numeric, got {_lower} ({type(_lower)}) and {_upper} ({type(_upper)}).")

        if _lower > _upper:
            raise ValueError(f"For each bound, lower bound must be smaller than upper bound, got {lower}, {upper})")

        if _upper - _lower < min_separation:
            mid = (_upper + _lower) / 2
            lower[i] = mid - min_separation / 2
            upper[i] = mid + min_separation / 2

    if shape == 0:
        return lower.item(), upper.item()

    if n_bounds == 1:
        lower = np.ones(shape, dtype=dtype) * lower.item()
        upper = np.ones(shape, dtype=dtype) * upper.item()

    return lower, upper


def clip_to_norm(array, clip):
    """Clips the examples of a 2-dimensional array to a given maximum norm.

    Parameters
    ----------
    array : np.ndarray
        Array to be clipped.  After clipping, all examples have a 2-norm of at most `clip`.

    clip : float
        Norm at which to clip each example

    Returns
    -------
    array : np.ndarray
        The clipped array.

    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input array must be a numpy array, got {type(array)}.")
    if array.ndim != 2:
        raise ValueError(f"input array must be 2-dimensional, got {array.ndim} dimensions.")
    if not isinstance(clip, Real):
        raise TypeError(f"Clip value must be numeric, got {type(clip)}.")
    if clip <= 0:
        raise ValueError(f"Clip value must be strictly positive, got {clip}.")

    norms = np.linalg.norm(array, axis=1) / clip
    norms[norms < 1] = 1

    return array / norms[:, np.newaxis]


def clip_to_bounds(array, bounds):
    """Clips the examples of a 2-dimensional array to given bounds.

    Parameters
    ----------
    array : np.ndarray
        Array to be clipped.  After clipping, all examples have a 2-norm of at most `clip`.

    bounds : tuple
        Tuple of bounds of the form (min, max) which the array is to be clipped to. `min` and `max` must be scalar,
        unless array is 2-dimensional.

    Returns
    -------
    array : np.ndarray
        The clipped array.

    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input array must be a numpy array, got {type(array)}.")

    lower, upper = check_bounds(bounds, np.size(bounds[0]), min_separation=0)
    clipped_array = array.copy()

    if np.allclose(lower, np.min(lower)) and np.allclose(upper, np.max(upper)):
        clipped_array = np.clip(clipped_array, np.min(lower), np.max(upper))
    else:
        if array.ndim != 2:
            raise ValueError(f"For non-scalar bounds, input array must be 2-dimensional. Got {array.ndim} dimensions.")

        for feature in range(array.shape[1]):
            clipped_array[:, feature] = np.clip(array[:, feature], lower[feature], upper[feature])

    return clipped_array


class DiffprivlibMixin:
    """Mixin for Diffprivlib models."""
    _check_bounds = staticmethod(check_bounds)
    _clip_to_norm = staticmethod(clip_to_norm)
    _clip_to_bounds = staticmethod(clip_to_bounds)
    _warn_unused_args = staticmethod(warn_unused_args)
