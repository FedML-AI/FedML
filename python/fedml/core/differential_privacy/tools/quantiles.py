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
Quantile functions with differential privacy
"""
import warnings

import numpy as np

from fedml.core.differential_privacy.accountant import BudgetAccountant
from fedml.core.differential_privacy.mechanisms import Exponential
from fedml.core.differential_privacy.utils import warn_unused_args, PrivacyLeakWarning
from fedml.core.differential_privacy.validation import clip_to_bounds, check_bounds
from fedml.core.differential_privacy.tools.utils import _wrap_axis


def quantile(array, quant, epsilon=1.0, bounds=None, axis=None, keepdims=False, accountant=None, **unused_args):
    r"""
    Compute the differentially private quantile of the array.

    Returns the specified quantile with differential privacy.  The quantile is calculated over the flattened array.
    Differential privacy is achieved with the :class:`.Exponential` mechanism, using the method first proposed by
    Smith, 2011.

    Paper link: https://dl.acm.org/doi/pdf/10.1145/1993636.1993743

    Parameters
    ----------
    array : array_like
        Array containing numbers whose quantile is sought.  If `array` is not an array, a conversion is attempted.

    quant : float or array-like
        Quantile or array of quantiles.  Each quantile must be in the unit interval [0, 1].  If quant is array-like,
        quantiles are returned over the flattened array.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.  Differential privacy is achieved over the entire output, with epsilon split
        evenly between each output value.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default, axis=None, will sum all of the elements of the input
        array.  If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single
        axis or all the axes as before.

    keepdims : bool, default: False
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `mean` method of sub-classes
        of `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    m : ndarray
        Returns a new array containing the quantile values.

    See Also
    --------
    numpy.quantile : Equivalent non-private method.

    percentile, median

    """
    warn_unused_args(unused_args)

    if bounds is None:
        warnings.warn("Bounds have not been specified and will be calculated on the data provided. This will "
                      "result in additional privacy leakage. To ensure differential privacy and no additional "
                      "privacy leakage, specify bounds for each dimension.", PrivacyLeakWarning)
        bounds = (np.min(array), np.max(array))

    quant = np.ravel(quant)

    if np.any(quant < 0) or np.any(quant > 1):
        raise ValueError("Quantiles must be in the unit interval [0, 1].")

    if len(quant) > 1:
        return np.array([quantile(array, q_i, epsilon=epsilon / len(quant), bounds=bounds, axis=axis, keepdims=keepdims,
                                  accountant=accountant) for q_i in quant])

    # Dealing with a single quant from now on
    quant = quant.item()

    if axis is not None or keepdims:
        return _wrap_axis(quantile, array, quant=quant, epsilon=epsilon, bounds=bounds, axis=axis, keepdims=keepdims,
                          accountant=accountant)

    # Dealing with a scalar output from now on
    bounds = check_bounds(bounds, shape=0, min_separation=1e-5)

    accountant = BudgetAccountant.load_default(accountant)
    accountant.check(epsilon, 0)

    # Let's ravel array to be single-dimensional
    array = clip_to_bounds(np.ravel(array), bounds)

    k = array.size
    array = np.append(array, list(bounds))
    array.sort()

    interval_sizes = np.diff(array)

    # Todo: Need to find a way to do this in a differentially private way
    if np.isnan(interval_sizes).any():
        return np.nan

    mech = Exponential(epsilon=epsilon, sensitivity=1, utility=list(-np.abs(np.arange(0, k + 1) - quant * k)),
                       measure=list(interval_sizes))
    idx = mech.randomise()
    output = mech._rng.random() * (array[idx+1] - array[idx]) + array[idx]

    accountant.spend(epsilon, 0)

    return output


def percentile(array, percent, epsilon=1.0, bounds=None, axis=None, keepdims=False, accountant=None, **unused_args):
    r"""
    Compute the differentially private percentile of the array.

    This method calls :obj:`.quantile`, where quantile = percentile / 100.

    Parameters
    ----------
    array : array_like
        Array containing numbers whose percentile is sought.  If `array` is not an array, a conversion is attempted.

    percent : float or array-like
        Percentile or list of percentiles sought.  Each percentile must be in [0, 100].  If percent is array-like,
        percentiles are returned over the flattened array.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.  Differential privacy is achieved over the entire output, with epsilon split
        evenly between each output value.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default, axis=None, will sum all of the elements of the input
        array.  If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single
        axis or all the axes as before.

    keepdims : bool, default: False
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `mean` method of sub-classes
        of `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    m : ndarray
        Returns a new array containing the percentile values.

    See Also
    --------
    numpy.percentile : Equivalent non-private method.

    quantile, median

    """
    warn_unused_args(unused_args)

    quant = np.asarray(percent) / 100

    if np.any(quant < 0) or np.any(quant > 1):
        raise ValueError("Percentiles must be between 0 and 100 inclusive")

    return quantile(array, quant, epsilon=epsilon, bounds=bounds, axis=axis, keepdims=keepdims, accountant=accountant)


def median(array, epsilon=1.0, bounds=None, axis=None, keepdims=False, accountant=None, **unused_args):
    r"""
    Compute the differentially private median of the array.

    Returns the median with differential privacy.  The median is calculated over each axis, or the flattened array
    if an axis is not provided.  This method calls the :obj:`.quantile` method, for the 0.5 quantile.

    Parameters
    ----------
    array : array_like
        Array containing numbers whose median is sought.  If `array` is not an array, a conversion is attempted.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.  Differential privacy is achieved over the entire output, with epsilon split
        evenly between each output value.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default, axis=None, will sum all of the elements of the input
        array.  If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single
        axis or all the axes as before.

    keepdims : bool, default: False
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `mean` method of sub-classes
        of `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    m : ndarray
        Returns a new array containing the median values.

    See Also
    --------
    numpy.median : Equivalent non-private method.

    quantile, percentile

    """
    warn_unused_args(unused_args)

    return quantile(array, 0.5, epsilon=epsilon, bounds=bounds, axis=axis, keepdims=keepdims, accountant=accountant)
