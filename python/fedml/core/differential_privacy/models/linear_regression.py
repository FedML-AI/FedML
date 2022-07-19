# MIT License
#
# Copyright (C) IBM Corporation 2019
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
#
#
# New BSD License
#
# Copyright (c) 2007–2019 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#      disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#      following disclaimer in the documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of its contributors may be used to endorse or
#      promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
Linear Regression with differential privacy
"""
import warnings

import numpy as np
import sklearn.linear_model as sk_lr
from scipy.optimize import minimize
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES

from fedml.core.differential_privacy.accountant import BudgetAccountant
from fedml.core.differential_privacy.mechanisms import Laplace, LaplaceFolded
from fedml.core.differential_privacy.tools import mean
from fedml.core.differential_privacy.utils import warn_unused_args, PrivacyLeakWarning
from fedml.core.differential_privacy.validation import check_bounds, clip_to_bounds, DiffprivlibMixin


# noinspection PyPep8Naming
def _preprocess_data(X, y, fit_intercept, epsilon=1.0, bounds_X=None, bounds_y=None, copy=True, check_input=True,
                     **unused_args):
    warn_unused_args(unused_args)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=False, dtype=FLOAT_DTYPES)
    elif copy:
        X = X.copy(order='K')

    y = np.asarray(y, dtype=X.dtype)
    X_scale = np.ones(X.shape[1], dtype=X.dtype)

    if fit_intercept:
        bounds_X = check_bounds(bounds_X, X.shape[1])
        bounds_y = check_bounds(bounds_y, y.shape[1] if y.ndim > 1 else 1)

        X = clip_to_bounds(X, bounds_X)
        y = clip_to_bounds(y, bounds_y)

        X_offset = mean(X, axis=0, bounds=bounds_X, epsilon=epsilon, accountant=BudgetAccountant())
        X -= X_offset
        y_offset = mean(y, axis=0, bounds=bounds_y, epsilon=epsilon, accountant=BudgetAccountant())
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


def _construct_regression_obj(X, y, bounds_X, bounds_y, epsilon, alpha):
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_features = X.shape[1]
    n_targets = y.shape[1]

    local_epsilon = epsilon / (1 + n_targets * n_features + n_features * (n_features + 1) / 2)
    coefs = ((y ** 2).sum(axis=0), np.einsum('ij,ik->jk', X, y), np.einsum('ij,ik', X, X))

    del X, y

    def get_max_sensitivity(y_lower, y_upper, x_lower, x_upper):
        corners = [y_lower * x_lower, y_lower * x_upper, y_upper * x_lower, y_upper * x_upper]
        return np.max(corners) - np.min(corners)

    # Randomise 0th-degree monomial coefficients
    mono_coef_0 = np.zeros(n_targets)

    for i in range(n_targets):
        sensitivity = np.abs([bounds_y[0][i], bounds_y[1][i]]).max() ** 2
        mech = LaplaceFolded(epsilon=local_epsilon, sensitivity=sensitivity, lower=0, upper=float("inf"))
        mono_coef_0[i] = mech.randomise(coefs[0][i])

    # Randomise 1st-degree monomial coefficients
    mono_coef_1 = np.zeros((n_features, n_targets))

    for i in range(n_targets):
        for j in range(n_features):
            sensitivity = get_max_sensitivity(bounds_y[0][i], bounds_y[1][i], bounds_X[0][j], bounds_X[1][j])
            mech = Laplace(epsilon=local_epsilon, sensitivity=sensitivity)
            mono_coef_1[j, i] = mech.randomise(coefs[1][j, i])

    # Randomise 2nd-degree monomial coefficients
    mono_coef_2 = np.zeros((n_features, n_features))

    for i in range(n_features):
        sensitivity = np.max(np.abs([bounds_X[0][i], bounds_X[0][i]])) ** 2
        mech = LaplaceFolded(epsilon=local_epsilon, sensitivity=sensitivity, lower=0, upper=float("inf"))
        mono_coef_2[i, i] = mech.randomise(coefs[2][i, i])

        for j in range(i + 1, n_features):
            sensitivity = get_max_sensitivity(bounds_X[0][i], bounds_X[1][i], bounds_X[0][j], bounds_X[1][j])
            mech = Laplace(epsilon=local_epsilon, sensitivity=sensitivity)
            mono_coef_2[i, j] = mech.randomise(coefs[2][i, j])
            mono_coef_2[j, i] = mono_coef_2[i, j]  # Enforce symmetry

    del coefs
    noisy_coefs = (mono_coef_0, mono_coef_1, mono_coef_2)

    def obj(idx):
        def inner_obj(omega):
            func = noisy_coefs[0][idx]
            func -= 2 * np.dot(noisy_coefs[1][:, idx], omega)
            func += np.multiply(noisy_coefs[2], np.tensordot(omega, omega, axes=0)).sum()
            func += alpha * (omega ** 2).sum()

            grad = - 2 * noisy_coefs[1][:, idx] + 2 * np.matmul(noisy_coefs[2], omega) + 2 * omega * alpha

            return func, grad

        return inner_obj

    output = tuple(obj(i) for i in range(n_targets))

    return output, noisy_coefs


# noinspection PyPep8Naming,PyAttributeOutsideInit
class LinearRegression(sk_lr.LinearRegression, DiffprivlibMixin):
    r"""
    Ordinary least squares Linear Regression with differential privacy.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp) to minimize the residual sum of squares
    between the observed targets in the dataset, and the targets predicted by the linear approximation.  Differential
    privacy is guaranteed with respect to the training sample.

    Differential privacy is achieved by adding noise to the coefficients of the objective function, taking inspiration
    from [ZZX12]_.

    Parameters
    ----------
    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds_X:  tuple
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    bounds_y : tuple
        Same as `bounds_X`, but for the training label set `y`.

    fit_intercept : bool, default: True
        Whether to calculate the intercept for this model.  If set to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default: True
        If True, X will be copied; else, it may be overwritten.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.  If multiple targets are passed during the fit (y 2D),
        this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of
        length n_features.

    intercept_ : float or array of shape of (n_targets,)
        Independent term in the linear model.  Set to 0.0 if `fit_intercept = False`.

    References
    ----------
    .. [ZZX12] Zhang, Jun, Zhenjie Zhang, Xiaokui Xiao, Yin Yang, and Marianne Winslett. "Functional mechanism:
        regression analysis under differential privacy." arXiv preprint arXiv:1208.0219 (2012).

    """
    def __init__(self, *, epsilon=1.0, bounds_X=None, bounds_y=None, fit_intercept=True, copy_X=True, accountant=None,
                 **unused_args):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=None)

        self.epsilon = epsilon
        self.bounds_X = bounds_X
        self.bounds_y = bounds_y
        self.accountant = BudgetAccountant.load_default(accountant)

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data

        y : array_like, shape (n_samples, n_targets)
            Target values.  Will be cast to X's dtype if necessary

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self : returns an instance of self.

        """
        self.accountant.check(self.epsilon, 0)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        X, y = self._validate_data(X, y, accept_sparse=False, y_numeric=True, multi_output=True)

        if self.bounds_X is None or self.bounds_y is None:
            warnings.warn(
                "Bounds parameters haven't been specified, so falling back to determining bounds from the "
                "data.\n"
                "This will result in additional privacy leakage. To ensure differential privacy with no "
                "additional privacy loss, specify `bounds_X` and `bounds_y`.",
                PrivacyLeakWarning)

            if self.bounds_X is None:
                self.bounds_X = (np.min(X, axis=0), np.max(X, axis=0))
            if self.bounds_y is None:
                self.bounds_y = (np.min(y, axis=0), np.max(y, axis=0))

        self.bounds_X = self._check_bounds(self.bounds_X, X.shape[1])
        self.bounds_y = self._check_bounds(self.bounds_y, y.shape[1] if y.ndim > 1 else 1)

        n_features = X.shape[1]
        n_targets = y.shape[1] if y.ndim > 1 else 1
        epsilon_intercept_scale = 1 / (n_features + 1) if self.fit_intercept else 0

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, bounds_X=self.bounds_X, bounds_y=self.bounds_y,
            epsilon=self.epsilon * epsilon_intercept_scale, copy=self.copy_X)

        bounds_X = (self.bounds_X[0] - X_offset, self.bounds_X[1] - X_offset)
        bounds_y = (self.bounds_y[0] - y_offset, self.bounds_y[1] - y_offset)

        objs, obj_coefs = _construct_regression_obj(
            X, y, bounds_X, bounds_y, epsilon=self.epsilon * (1 - epsilon_intercept_scale), alpha=0)
        coef = np.zeros((n_features, n_targets))

        for i, obj in enumerate(objs):
            opt_result = minimize(obj, np.zeros(n_features), jac=True)
            coef[:, i] = opt_result.x

        self.coef_ = coef.T
        self._obj_coefs = obj_coefs

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)

        self.accountant.spend(self.epsilon, 0)

        return self

    _preprocess_data = staticmethod(_preprocess_data)
