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
"""
K-means clustering algorithm satisfying differential privacy.
"""
import warnings

import numpy as np
import sklearn.cluster as sk_cluster

from fedml.core.differential_privacy.accountant import BudgetAccountant
from fedml.core.differential_privacy.mechanisms import LaplaceBoundedDomain, GeometricFolded
from fedml.core.differential_privacy.utils import PrivacyLeakWarning
from fedml.core.differential_privacy.validation import DiffprivlibMixin


class KMeans(sk_cluster.KMeans, DiffprivlibMixin):
    r"""K-Means clustering with differential privacy.

    Implements the DPLloyd approach presented in [SCL16]_, leveraging the :class:`sklearn.cluster.KMeans` class for full
    integration with Scikit Learn.

    Parameters
    ----------
    n_clusters : int, default: 8
        The number of clusters to form as well as the number of centroids to generate.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds:  tuple, optional
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.  If the algorithm stops before fully converging, these will not be consistent
        with ``labels_``.

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.

    References
    ----------
    .. [SCL16] Su, Dong, Jianneng Cao, Ninghui Li, Elisa Bertino, and Hongxia Jin. "Differentially private k-means
        clustering." In Proceedings of the sixth ACM conference on data and application security and privacy, pp. 26-37.
        ACM, 2016.

    """

    def __init__(self, n_clusters=8, *, epsilon=1.0, bounds=None, accountant=None, **unused_args):
        super().__init__(n_clusters=n_clusters)

        self.epsilon = epsilon
        self.bounds = bounds
        self.accountant = BudgetAccountant.load_default(accountant)

        self._warn_unused_args(unused_args)

        self.cluster_centers_ = None
        self.bounds_processed = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self._n_threads = 1

    def fit(self, X, y=None, sample_weight=None):
        """Computes k-means clustering with differential privacy.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            not used, present here for API consistency by convention.

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self : class

        """
        self.accountant.check(self.epsilon, 0)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        del y

        X = self._validate_data(X, accept_sparse=False, dtype=[np.float64, np.float32])
        n_samples, n_dims = X.shape

        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}")

        iters = self._calc_iters(n_dims, n_samples)

        if self.bounds is None:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided.  This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify `bounds` for each dimension.", PrivacyLeakWarning)
            self.bounds = (np.min(X, axis=0), np.max(X, axis=0))

        self.bounds = self._check_bounds(self.bounds, n_dims, min_separation=1e-5)
        X = self._clip_to_bounds(X, self.bounds)

        centers = self._init_centers(n_dims)
        labels = None
        distances = None

        # Run _update_centers first to ensure consistency of `labels` and `centers`, since convergence unlikely
        for _ in range(-1, iters):
            if labels is not None:
                centers = self._update_centers(X, centers=centers, labels=labels, dims=n_dims, total_iters=iters)

            distances, labels = self._distances_labels(X, centers)

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = distances[np.arange(len(labels)), labels].sum()
        self.n_iter_ = iters

        self.accountant.spend(self.epsilon, 0)

        return self

    def _init_centers(self, dims):
        if self.bounds_processed is None:
            bounds_processed = np.zeros(shape=(dims, 2))

            for dim in range(dims):
                lower = self.bounds[0][dim]
                upper = self.bounds[1][dim]

                bounds_processed[dim, :] = [upper - lower, lower]

            self.bounds_processed = bounds_processed

        cluster_proximity = np.min(self.bounds_processed[:, 0]) / 2.0

        while cluster_proximity > 0:
            centers = np.zeros(shape=(self.n_clusters, dims))
            cluster, retry = 0, 0

            while retry < 100:
                if cluster >= self.n_clusters:
                    break

                temp_center = np.random.random(dims) * (self.bounds_processed[:, 0] - 2 * cluster_proximity) + \
                    self.bounds_processed[:, 1] + cluster_proximity

                if cluster == 0:
                    centers[0, :] = temp_center
                    cluster += 1
                    continue

                min_distance = ((centers[:cluster, :] - temp_center) ** 2).sum(axis=1).min()

                if np.sqrt(min_distance) >= 2 * cluster_proximity:
                    centers[cluster, :] = temp_center
                    cluster += 1
                    retry = 0
                else:
                    retry += 1

            if cluster >= self.n_clusters:
                return centers

            cluster_proximity /= 2.0

        return None

    def _distances_labels(self, X, centers):
        distances = np.zeros((X.shape[0], self.n_clusters))

        for cluster in range(self.n_clusters):
            distances[:, cluster] = ((X - centers[cluster, :]) ** 2).sum(axis=1)

        labels = np.argmin(distances, axis=1)
        return distances, labels

    def _update_centers(self, X, centers, labels, dims, total_iters):
        """Updates the centers of the KMeans algorithm for the current iteration, while satisfying differential
        privacy.

        Differential privacy is satisfied by adding (integer-valued, using :class:`.GeometricFolded`) random noise to
        the count of nearest neighbours to the previous cluster centers, and adding (real-valued, using
        :class:`.LaplaceBoundedDomain`) random noise to the sum of values per dimension.

        """
        epsilon_0, epsilon_i = self._split_epsilon(dims, total_iters)
        geometric_mech = GeometricFolded(epsilon=epsilon_0, sensitivity=1, lower=0.5, upper=float("inf"))

        for cluster in range(self.n_clusters):
            if cluster not in labels:
                continue

            cluster_count = sum(labels == cluster)
            noisy_count = geometric_mech.randomise(cluster_count)

            cluster_sum = np.sum(X[labels == cluster], axis=0)
            noisy_sum = np.zeros_like(cluster_sum)

            for i in range(dims):
                laplace_mech = LaplaceBoundedDomain(epsilon=epsilon_i,
                                                    sensitivity=self.bounds[1][i] - self.bounds[0][i],
                                                    lower=noisy_count * self.bounds[0][i],
                                                    upper=noisy_count * self.bounds[1][i])
                noisy_sum[i] = laplace_mech.randomise(cluster_sum[i])

            centers[cluster, :] = noisy_sum / noisy_count

        return centers

    def _split_epsilon(self, dims, total_iters, rho=0.225):
        """Split epsilon between sum perturbation and count perturbation, as proposed by Su et al.

        Parameters
        ----------
        dims : int
            Number of dimensions to split `epsilon` across.

        total_iters : int
            Total number of iterations to split `epsilon` across.

        rho : float, default: 0.225
            Coordinate normalisation factor.

        Returns
        -------
        epsilon_0 : float
            The epsilon value for satisfying differential privacy on the count of a cluster.

        epsilon_i : float
            The epsilon value for satisfying differential privacy on each dimension of the center of a cluster.

        """
        epsilon_i = 1
        epsilon_0 = np.cbrt(4 * dims * rho ** 2)

        normaliser = self.epsilon / total_iters / (epsilon_i * dims + epsilon_0)

        return epsilon_i * normaliser, epsilon_0 * normaliser

    def _calc_iters(self, n_dims, n_samples, rho=0.225):
        """Calculate the number of iterations to allow for the KMeans algorithm."""

        epsilon_m = np.sqrt(500 * (self.n_clusters ** 3) / (n_samples ** 2) *
                            (n_dims + np.cbrt(4 * n_dims * (rho ** 2))) ** 3)

        iters = max(min(self.epsilon / epsilon_m, 7), 2)

        return int(iters)
