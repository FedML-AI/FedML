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
Logistic Regression classifier satisfying differential privacy.
"""
import numbers
import warnings

import numpy as np
from joblib import delayed, Parallel
from scipy import optimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.extmath import safe_sparse_dot, log_logistic
from sklearn.utils.multiclass import check_classification_targets

from fedml.core.differential_privacy.accountant import BudgetAccountant
from fedml.core.differential_privacy.mechanisms import Vector
from fedml.core.differential_privacy.utils import PrivacyLeakWarning, DiffprivlibCompatibilityWarning, warn_unused_args
from fedml.core.differential_privacy.validation import DiffprivlibMixin


class LogisticRegression(linear_model.LogisticRegression, DiffprivlibMixin):
    r"""Logistic Regression (aka logit, MaxEnt) classifier with differential privacy.

    This class implements regularised logistic regression using :ref:`Scipy's L-BFGS-B algorithm
    <scipy:optimize.minimize-lbfgsb>`.  :math:`\epsilon`-Differential privacy is achieved relative to the maximum norm
    of the data, as determined by `data_norm`, by the :class:`.Vector` mechanism, which adds a Laplace-distributed
    random vector to the objective.  Adapted from the work presented in [CMS11]_.

    This class is a child of :obj:`sklearn.linear_model.LogisticRegression`, with amendments to allow for the
    implementation of differential privacy.  Some parameters of `Scikit Learn`'s model have therefore had to be fixed,
    including:

        - The only permitted solver is 'lbfgs'.  Specifying the ``solver`` option will result in a warning.
        - Consequently, the only permitted penalty is 'l2'.  Specifying the ``penalty`` option will result in a warning.
        - In the multiclass case, only the one-vs-rest (OvR) scheme is permitted.  Specifying the ``multi_class`` option
          will result in a warning.

    Parameters
    ----------
    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    data_norm : float, optional
        The max l2 norm of any row of the data.  This defines the spread of data that will be protected by
        differential privacy.

        If not specified, the max norm is taken from the data when ``.fit()`` is first called, but will result in a
        :class:`.PrivacyLeakWarning`, as it reveals information about the data.  To preserve differential privacy fully,
        `data_norm` should be selected independently of the data, i.e. with domain knowledge.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.  Like in support vector machines, smaller values
        specify stronger regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

    max_iter : int, default: 100
        Maximum number of iterations taken for the solver to converge.  For smaller `epsilon` (more noise), `max_iter`
        may need to be increased.

    verbose : int, default: 0
        Set to any positive number for verbosity.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit as initialization, otherwise, just erase
        the previous solution.

    n_jobs : int, optional
        Number of CPU cores used when parallelising over classes.  ``None`` means 1 unless in a context. ``-1`` means
        using all processors.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Attributes
    ----------
    classes_ : array, shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : array, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.

    intercept_ : array, shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero. `intercept_` is of shape (1,) when the
        given problem is binary.

    n_iter_ : array, shape (n_classes,) or (1, )
        Actual number of iterations for all classes.  If binary, it returns only 1 element.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from fedml.core.differential_privacy.models import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(data_norm=12, epsilon=2).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[7.35362932e-01, 2.16667422e-14, 2.64637068e-01],
           [9.08384378e-01, 3.47767052e-13, 9.16156215e-02]])
    >>> clf.score(X, y)
    0.5266666666666666

    See also
    --------
    sklearn.linear_model.LogisticRegression : The implementation of logistic regression in scikit-learn, upon which this
        implementation is built.
    .Vector : The mechanism used by the model to achieve differential privacy.

    References
    ----------
    .. [CMS11] Chaudhuri, Kamalika, Claire Monteleoni, and Anand D. Sarwate. "Differentially private empirical risk
        minimization." Journal of Machine Learning Research 12, no. Mar (2011): 1069-1109.

    """

    def __init__(self, *, epsilon=1.0, data_norm=None, tol=1e-4, C=1.0, fit_intercept=True, max_iter=100, verbose=0,
                 warm_start=False, n_jobs=None, accountant=None, **unused_args):
        super().__init__(penalty='l2', dual=False, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=1.0,
                         class_weight=None, random_state=None, solver='lbfgs', max_iter=max_iter, multi_class='ovr',
                         verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        self.epsilon = epsilon
        self.data_norm = data_norm
        self.classes_ = None
        self.accountant = BudgetAccountant.load_default(accountant)

        self._warn_unused_args(unused_args)

    # noinspection PyAttributeOutsideInit
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self : class

        """
        self.accountant.check(self.epsilon, 0)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        if not isinstance(self.C, numbers.Real) or self.C < 0:
            raise ValueError(f"Penalty term must be positive; got (C={self.C})")
        if not isinstance(self.max_iter, numbers.Integral) or self.max_iter < 0:
            raise ValueError(f"Maximum number of iteration must be positive; got (max_iter={self.max_iter})")
        if not isinstance(self.tol, numbers.Real) or self.tol < 0:
            raise ValueError(f"Tolerance for stopping criteria must be positive; got (tol={self.tol})")

        solver = _check_solver(self.solver, self.penalty, self.dual)
        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=float, order="C",
                                   accept_large_sparse=solver != 'liblinear')
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        _, n_features = X.shape

        if self.data_norm is None:
            warnings.warn("Data norm has not been specified and will be calculated on the data provided.  This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify `data_norm` at initialisation.", PrivacyLeakWarning)
            self.data_norm = np.linalg.norm(X, axis=1).max()

        X = self._clip_to_norm(X, self.data_norm)

        self.multi_class = _check_multi_class(self.multi_class, solver, len(self.classes_))

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes in the data, but the data contains only "
                             f"one class: {classes_[0]}")

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef, self.intercept_[:, np.newaxis], axis=1)

        self.coef_ = []
        self.intercept_ = np.zeros(n_classes)

        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(_logistic_regression_path)

        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='processes')(
            path_func(X, y, epsilon=self.epsilon / n_classes, data_norm=self.data_norm, pos_class=class_, Cs=[self.C],
                      fit_intercept=self.fit_intercept, max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
                      coef=warm_start_coef_, check_input=False)
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef))

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        self.coef_ = np.asarray(fold_coefs_)
        self.coef_ = self.coef_.reshape(n_classes, n_features + int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        self.accountant.spend(self.epsilon, 0)

        return self


def _logistic_regression_path(X, y, epsilon, data_norm, pos_class=None, Cs=10, fit_intercept=True, max_iter=100,
                              tol=1e-4, verbose=0, coef=None, check_input=True, **unused_args):
    """Compute a Logistic Regression model with differential privacy for a list of regularization parameters.  Takes
    inspiration from ``_logistic_regression_path`` in scikit-learn, specified to the LBFGS solver and one-vs-rest
    multi class fitting.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    epsilon : float
        Privacy parameter for differential privacy.

    data_norm : float
        Max norm of the data for which differential privacy is satisfied.

    pos_class : int, optional
        The class with respect to which we perform a one-vs-all fit.  If None, then it is assumed that the given problem
        is binary.

    Cs : int | array-like, shape (n_cs,), default: 10
        List of values for the regularization parameter or integer specifying the number of regularization parameters
        that should be used.  In this case, the parameters will be chosen in a logarithmic scale between 1e-4 and 1e4.

    fit_intercept : bool, default: True
        Whether to fit an intercept for the model.  In this case the shape of the returned array is
        (n_cs, n_features + 1).

    max_iter : int, default: 100
        Maximum number of iterations for the solver.

    tol : float, default: 1e-4
        Stopping criterion.  For the newton-cg and lbfgs solvers, the iteration will stop when ``max{|g_i | i = 1,
        ..., n} <= tol`` where ``g_i`` is the i-th component of the gradient.

    verbose : int, default: 0
        For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.

    coef : array-like, shape (n_features,), optional
        Initialization value for coefficients of logistic regression.  Useless for liblinear solver.

    check_input : bool, default: True
        If False, the input arrays X and y will not be checked.

    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model.  If fit_intercept is set to True then the second
        dimension will be n_features + 1, where the last item represents the intercept.  For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs, n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array, shape (n_cs,)
        Actual number of iteration for each Cs.

    """
    warn_unused_args(unused_args)

    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, int(Cs))

    solver = 'lbfgs'

    # Data norm increases if intercept is included
    if fit_intercept:
        data_norm = np.sqrt(data_norm ** 2 + 1)

    # Pre-processing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64, accept_large_sparse=solver != 'liblinear')
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape

    classes = np.unique(y)

    if pos_class is None:
        if classes.size > 2:
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # For doing a ovr, we need to mask the labels first.
    output_vec = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
    mask = (y == pos_class)
    y_bin = np.ones(y.shape, dtype=X.dtype)
    y_bin[~mask] = -1.
    # for compute_class_weight

    if coef is not None:
        # it must work both giving the bias term and not
        if coef.size not in (n_features, output_vec.size):
            raise ValueError(f"Initialization coef is of shape {coef.size}, expected shape {n_features} or "
                             f"{output_vec.size}")
        output_vec[:coef.size] = coef

    target = y_bin

    coefs = []
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        vector_mech = Vector(epsilon=epsilon, dimension=n_features + int(fit_intercept), alpha=1. / C,
                             function_sensitivity=0.25, data_sensitivity=data_norm)
        noisy_logistic_loss = vector_mech.randomise(_logistic_loss_and_grad)

        iprint = [-1, 50, 1, 100, 101][np.searchsorted(np.array([0, 1, 2, 3]), verbose)]
        output_vec, _, info = optimize.fmin_l_bfgs_b(noisy_logistic_loss, output_vec, fprime=None,
                                                     args=(X, target, 1. / C, sample_weight), iprint=iprint, pgtol=tol,
                                                     maxiter=max_iter)
        if info["warnflag"] == 1:
            warnings.warn("lbfgs failed to converge. Increase the number of iterations.", ConvergenceWarning)

        coefs.append(output_vec.copy())

        n_iter[i] = info['nit']

    return np.array(coefs), np.array(Cs), n_iter


def _logistic_loss_and_grad(w, X, y, alpha, sample_weight=None):
    """Computes the logistic loss and gradient.

    Parameters
    ----------
    w : ndarray of shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Array of labels.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,), default=None
        Array of weights that are assigned to individual samples.  If not provided, then each sample is given unit
        weight.

    Returns
    -------
    out : float
        Logistic loss.

    grad : ndarray of shape (n_features,) or (n_features + 1,)
        Logistic gradient.

    """
    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    w, c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(sample_weight * log_logistic(yz)) + 0.5 * alpha * np.dot(w, w)

    z = expit(yz)
    z0 = sample_weight * (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()
    return out, grad


def _intercept_dot(w, X, y):
    """Computes y * np.dot(X, w).
    It takes into consideration if the intercept should be fit or not.

    Parameters
    ----------
    w : ndarray of shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Array of labels.

    Returns
    -------
    w : ndarray of shape (n_features,)
        Coefficient vector without the intercept weight (w[-1]) if the intercept should be fit. Unchanged otherwise.

    c : float
        The intercept.

    yz : float
        y * np.dot(X, w).

    """
    c = 0.0
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = safe_sparse_dot(X, w) + c
    yz = y * z
    return w, c, yz


def _check_solver(solver, penalty, dual):
    if solver != 'lbfgs':
        warnings.warn("For diffprivlib, solver must be 'lbfgs'.", DiffprivlibCompatibilityWarning)
        solver = 'lbfgs'

    if penalty != 'l2':
        raise ValueError(f"Solver {solver} supports only l2 penalties, got {penalty} penalty.")
    if dual:
        raise ValueError(f"Solver {solver} supports only dual=False, got dual={dual}")
    return solver


def _check_multi_class(multi_class, solver, n_classes):
    del solver, n_classes

    if multi_class != 'ovr':
        warnings.warn(f"For diffprivlib, multi_class must be 'ovr', got {multi_class}.",
                      DiffprivlibCompatibilityWarning)
        multi_class = 'ovr'

    return multi_class
