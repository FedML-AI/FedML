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
Utilities for use in machine learning models
"""
import warnings
from numbers import Integral

import numpy as np
from scipy.linalg import null_space

from fedml.core.differential_privacy.mechanisms import LaplaceBoundedDomain, Bingham
from fedml.core.differential_privacy.utils import PrivacyLeakWarning


def covariance_eig(array, epsilon=1.0, norm=None, dims=None, eigvals_only=False):
    r"""
    Return the eigenvalues and eigenvectors of the covariance matrix of `array`, satisfying differential privacy.

    Paper link: http://papers.nips.cc/paper/9567-differentially-private-covariance-estimation.pdf

    Parameters
    ----------
    array : array-like, shape (n_samples, n_features)
        Matrix for which the covariance matrix is sought.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    norm : float, optional
        The max l2 norm of any row of the input array.  This defines the spread of data that will be protected by
        differential privacy.

        If not specified, the max norm is taken from the data, but will result in a :class:`.PrivacyLeakWarning`, as it
        reveals information about the data.  To preserve differential privacy fully, `norm` should be selected
        independently of the data, i.e. with domain knowledge.

    dims : int, optional
        Number of eigenvectors to return.  If `None`, return all eigenvectors.

    eigvals_only : bool, default: False
        Only return the eigenvalue estimates.  If True, all the privacy budget is spent on estimating the eigenvalues.

    Returns
    -------
    w : (n_features) array
        The eigenvalues, each repeated according to its multiplicity.

    v : (n_features, dims) array
        The normalized (unit "length") eigenvectors, such that the column ``v[:,i]`` is the eigenvector corresponding to
        the eigenvalue ``w[i]``.

    """

    n_features = array.shape[1]
    dims = n_features if dims is None else min(dims, n_features)
    if not isinstance(dims, Integral):
        raise TypeError(f"Number of requested dimensions must be integer-valued, got {type(dims)}")
    if dims < 0:
        raise ValueError(f"Number of requested dimensions must be non-negative, got {dims}")

    max_norm = np.linalg.norm(array, axis=1).max()
    if norm is None:
        warnings.warn("Data norm has not been specified and will be calculated on the data provided.  This will result "
                      "in additional privacy leakage. To ensure differential privacy and no additional privacy "
                      "leakage, specify `data_norm` at initialisation.", PrivacyLeakWarning)
        norm = max_norm
    elif max_norm > norm and not np.isclose(max_norm, norm):
        raise ValueError(f"Rows of input array must have l2 norm of at most {norm}, got {max_norm}")

    cov = array.T.dot(array) / (norm ** 2)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    epsilon_0 = epsilon if eigvals_only else epsilon / (dims + (dims != n_features))

    mech_eigvals = LaplaceBoundedDomain(epsilon=epsilon_0, lower=0, upper=float("inf"), sensitivity=2)
    noisy_eigvals = np.array([mech_eigvals.randomise(eigval) for eigval in eigvals]) * (norm ** 2)

    if eigvals_only:
        return noisy_eigvals

    # When estimating all eigenvectors, we don't need to spend budget for the dth vector
    epsilon_i = epsilon / (dims + (dims != n_features))
    cov_i = cov
    proj_i = np.eye(n_features)

    theta = np.zeros((0, n_features))
    mech_cov = Bingham(epsilon=epsilon_i)

    for _ in range(dims):
        if cov_i.size > 1:
            u_i = mech_cov.randomise(cov_i)
        else:
            u_i = np.ones((1,))

        theta_i = proj_i.T.dot(u_i)
        theta = np.vstack((theta, theta_i))

        if cov_i.size > 1:
            proj_i = null_space(theta).T
            cov_i = proj_i.dot(cov).dot(proj_i.T)

    return noisy_eigvals, theta.T
