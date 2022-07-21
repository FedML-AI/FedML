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
Privacy budget accountant for differential privacy
"""
from numbers import Integral

import numpy as np

from fedml.core.differential_privacy.utils import Budget, BudgetError
from fedml.core.differential_privacy.validation import check_epsilon_delta


class BudgetAccountant:
    """Privacy budget accountant for differential privacy.

    This class creates a privacy budget accountant to track privacy spend across queries and other data accesses.  Once
    initialised, the BudgetAccountant stores each privacy spend and iteratively updates the total budget spend, raising
    an error when the budget ceiling (if specified) is exceeded.  The accountant can be initialised without any maximum
    budget, to enable users track the total privacy spend of their actions without hindrance.

    Diffprivlib functions can make use of a BudgetAccountant in three different ways (see examples for more details):

        - Passed as an ``accountant`` parameter to the function (e.g., ``mean(..., accountant=acc)``)
        - Set as the default using the ``set_default()`` method (all subsequent diffprivlib functions will use the
          accountant by default)
        - As a context manager using a ``with`` statement (the accountant is used for that block of code)

    Implements the accountant rules as given in [KOV17]_.

    Parameters
    ----------
    epsilon : float, default: infinity
        Epsilon budget ceiling of the accountant.

    delta : float, default: 1.0
        Delta budget ceiling of the accountant.

    slack : float, default: 0.0
        Slack allowed in delta spend.  Greater slack may reduce the overall epsilon spend.

    spent_budget : list of tuples of the form (epsilon, delta), optional
        List of tuples of pre-existing budget spends.  Allows for a new accountant to be initialised with spends
        extracted from a previous instance.

    Attributes
    ----------
    epsilon : float
        Epsilon budget ceiling of the accountant.

    delta : float
        Delta budget ceiling of the accountant.

    slack : float
        The accountant's slack.  Can be modified at runtime, subject to the privacy budget not being exceeded.

    spent_budget : list of tuples of the form (epsilon, delta)
        The list of privacy spends recorded by the accountant.  Can be used in the initialisation of a new accountant.

    Examples
    --------

    A ``BudgetAccountant`` is typically passed to diffprivlib functions as an ``accountant`` parameter.  If ``epsilon``
    and ``delta`` are not set, the accountant has an infinite budget by default, allowing you to track privacy spend
    without imposing a hard limit.  By allowing a ``slack`` in the budget calculation, the overall epsilon privacy spend
    can be reduced (at the cost of extra delta spend).

    >>> import diffprivlib as dp
    >>> from numpy.random import random
    >>> X = random(100)
    >>> acc = dp.BudgetAccountant(epsilon=1.5, delta=0)
    >>> dp.tools.mean(X, bounds=(0, 1), accountant=acc)
    0.4547006207923884
    >>> acc.total()
    (epsilon=1.0, delta=0)
    >>> dp.tools.std(X, bounds=(0, 1), epsilon=0.25, accountant=acc)
    0.2630216611181259
    >>> acc.total()
    (epsilon=1.25, delta=0)

    >>> acc2 = dp.BudgetAccountant() # infinite budget
    >>> first_half = dp.tools.mean(X[:50], epsilon=0.25, bounds=(0, 1), accountant=acc2)
    >>> last_half = dp.tools.mean(X[50:], epsilon=0.25, bounds=(0, 1), accountant=acc2)
    >>> acc2.total()
    (epsilon=0.5, delta=0)
    >>> acc2.remaining()
    (epsilon=inf, delta=1.0)

    >>> acc3 = dp.BudgetAccountant(slack=1e-3)
    >>> for i in range(20):
    ...     dp.tools.mean(X, epsilon=0.05, bounds=(0, 1), accountant=acc3)
    >>> acc3.total() # Slack has reduced the epsilon spend by almost 25%
    (epsilon=0.7613352285668463, delta=0.001)

    Using ``set_default()``, an accountant is used by default in all diffprivlib functions in that script.  Accountants
    also act as context managers, allowing for use in a ``with`` statement.  Passing an accountant as a parameter
    overrides all other methods.

    >>> acc4 = dp.BudgetAccountant()
    >>> acc4.set_default()
    BudgetAccountant()
    >>> Y = random((100, 2)) - 0.5
    >>> clf = dp.models.PCA(1, centered=True, data_norm=1.4)
    >>> clf.fit(Y)
    PCA(accountant=BudgetAccountant(spent_budget=[(1.0, 0)]), centered=True, copy=True, data_norm=1.4, epsilon=1.0,
    n_components=1, random_state=None, bounds=None, whiten=False)
    >>> acc4.total()
    (epsilon=1.0, delta=0)

    >>> with dp.BudgetAccountant() as acc5:
    ...     dp.tools.mean(Y, bounds=(0, 1), epsilon=1/3)
    >>> acc5.total()
    (epsilon=0.3333333333333333, delta=0)

    References
    ----------
    .. [KOV17] Kairouz, Peter, Sewoong Oh, and Pramod Viswanath. "The composition theorem for differential privacy."
        IEEE Transactions on Information Theory 63.6 (2017): 4037-4049.

    """
    _default = None

    def __init__(self, epsilon=float("inf"), delta=1.0, slack=0.0, spent_budget=None):
        check_epsilon_delta(epsilon, delta)
        self.__epsilon = epsilon
        self.__min_epsilon = 0 if epsilon == float("inf") else epsilon * 1e-14
        self.__delta = delta
        self.__spent_budget = []
        self.slack = slack

        if spent_budget is not None:
            if not isinstance(spent_budget, list):
                raise TypeError("spent_budget must be a list")

            for _epsilon, _delta in spent_budget:
                self.spend(_epsilon, _delta)

    def __repr__(self, n_budget_max=5):
        params = []
        if self.epsilon != float("inf"):
            params.append(f"epsilon={self.epsilon}")

        if self.delta != 1:
            params.append(f"delta={self.delta}")

        if self.slack > 0:
            params.append(f"slack={self.slack}")

        if self.spent_budget:
            if len(self.spent_budget) > n_budget_max:
                params.append("spent_budget=" + str(self.spent_budget[:n_budget_max] + ["..."]).replace("'", ""))
            else:
                params.append("spent_budget=" + str(self.spent_budget))

        return "BudgetAccountant(" + ", ".join(params) + ")"

    def __enter__(self):
        self.old_default = self.pop_default()
        self.set_default()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pop_default()

        if self.old_default is not None:
            self.old_default.set_default()
        del self.old_default

    def __len__(self):
        return len(self.spent_budget)

    @property
    def slack(self):
        """Slack parameter for composition.
        """
        return self.__slack

    @slack.setter
    def slack(self, slack):
        if not 0 <= slack <= self.delta:
            raise ValueError(f"Slack must be between 0 and delta ({self.delta}), inclusive. Got {slack}.")

        epsilon_spent, delta_spent = self.total(slack=slack)

        if self.epsilon < epsilon_spent or self.delta < delta_spent:
            raise BudgetError(f"Privacy budget will be exceeded by changing slack to {slack}.")

        self.__slack = slack

    @property
    def spent_budget(self):
        """List of tuples of the form (epsilon, delta) of spent privacy budget.
        """
        return self.__spent_budget.copy()

    @property
    def epsilon(self):
        """Epsilon privacy ceiling of the accountant.
        """
        return self.__epsilon

    @property
    def delta(self):
        """Delta privacy ceiling of the accountant.
        """
        return self.__delta

    def total(self, spent_budget=None, slack=None):
        """Returns the total current privacy spend.

        `spent_budget` and `slack` can be specified as parameters, otherwise the class values will be used.

        Parameters
        ----------
        spent_budget : list of tuples of the form (epsilon, delta), optional
            List of tuples of budget spends.  If not provided, the accountant's spends will be used.

        slack : float, optional
            Slack in delta for composition.  If not provided, the accountant's slack will be used.

        Returns
        -------
        epsilon : float
            Total epsilon spend.

        delta : float
            Total delta spend.

        """
        if spent_budget is None:
            spent_budget = self.spent_budget
        else:
            for epsilon, delta in spent_budget:
                check_epsilon_delta(epsilon, delta)

        if slack is None:
            slack = self.slack
        elif not 0 <= slack <= self.delta:
            raise ValueError(f"Slack must be between 0 and delta ({self.delta}), inclusive. Got {slack}.")

        epsilon_sum, epsilon_exp_sum, epsilon_sq_sum = 0, 0, 0

        for epsilon, _ in spent_budget:
            epsilon_sum += epsilon
            epsilon_exp_sum += (1 - np.exp(-epsilon)) * epsilon / (1 + np.exp(-epsilon))
            epsilon_sq_sum += epsilon ** 2

        total_epsilon_naive = epsilon_sum
        total_delta = self.__total_delta_safe(spent_budget, slack)

        if slack == 0:
            return Budget(total_epsilon_naive, total_delta)

        total_epsilon_drv = epsilon_exp_sum + np.sqrt(2 * epsilon_sq_sum * np.log(1 / slack))
        total_epsilon_kov = epsilon_exp_sum + np.sqrt(2 * epsilon_sq_sum *
                                                      np.log(np.exp(1) + np.sqrt(epsilon_sq_sum) / slack))

        return Budget(min(total_epsilon_naive, total_epsilon_drv, total_epsilon_kov), total_delta)

    def check(self, epsilon, delta):
        """Checks if the provided (epsilon,delta) can be spent without exceeding the accountant's budget ceiling.

        Parameters
        ----------
        epsilon : float
            Epsilon budget spend to check.

        delta : float
            Delta budget spend to check.

        Returns
        -------
        bool
            True if the budget can be spent, otherwise a :class:`.BudgetError` is raised.

        Raises
        ------
        BudgetError
            If the specified budget spend will result in the budget ceiling being exceeded.

        """
        check_epsilon_delta(epsilon, delta)
        if self.epsilon == float("inf") and self.delta == 1:
            return True

        if 0 < epsilon < self.__min_epsilon:
            raise ValueError(f"Epsilon must be at least {self.__min_epsilon} if non-zero, got {epsilon}.")

        spent_budget = self.spent_budget + [(epsilon, delta)]

        if Budget(self.epsilon, self.delta) >= self.total(spent_budget=spent_budget):
            return True

        raise BudgetError(f"Privacy spend of ({epsilon},{delta}) not permissible; will exceed remaining privacy budget."
                          f"Use {self.__class__.__name__}.{self.remaining.__name__}() to check remaining budget.")

    def remaining(self, k=1):
        """Calculates the budget that remains to be spent.

        Calculates the privacy budget that can be spent on `k` queries.  Spending this budget on `k` queries will
        match the budget ceiling, assuming no floating point errors.

        Parameters
        ----------
        k : int, default: 1
            The number of queries for which to calculate the remaining budget.

        Returns
        -------
        epsilon : float
            Total epsilon spend remaining for `k` queries.

        delta : float
            Total delta spend remaining for `k` queries.

        """
        if not isinstance(k, Integral):
            raise TypeError(f"k must be integer-valued, got {type(k)}.")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}.")

        _, spent_delta = self.total()
        delta = 1 - ((1 - self.delta) / (1 - spent_delta)) ** (1 / k) if spent_delta < 1.0 else 1.0
        # delta = 1 - np.exp((np.log(1 - self.delta) - np.log(1 - spent_delta)) / k)

        lower = 0
        upper = self.epsilon
        old_interval_size = (upper - lower) * 2

        while old_interval_size > upper - lower:
            old_interval_size = upper - lower
            mid = (upper + lower) / 2

            spent_budget = self.spent_budget + [(mid, 0)] * k
            x_0, _ = self.total(spent_budget=spent_budget)

            if x_0 >= self.epsilon:
                upper = mid
            if x_0 <= self.epsilon:
                lower = mid

        epsilon = (upper + lower) / 2

        return Budget(epsilon, delta)

    def spend(self, epsilon, delta):
        """Spend the given privacy budget.

        Instructs the accountant to spend the given epsilon and delta privacy budget, while ensuring the target budget
        is not exceeded.

        Parameters
        ----------
        epsilon : float
            Epsilon privacy budget to spend.

        delta : float
            Delta privacy budget to spend.

        Returns
        -------
        self : BudgetAccountant

        """
        self.check(epsilon, delta)
        self.__spent_budget.append((epsilon, delta))
        return self

    @staticmethod
    def __total_delta_safe(spent_budget, slack):
        """
        Calculate total delta spend of `spent_budget`, with special consideration for floating point arithmetic.
        Should yield greater precision, especially for a large number of budget spends with very small delta.

        Parameters
        ----------
        spent_budget: list of tuples of the form (epsilon, delta)
            List of budget spends, for which the total delta spend is to be calculated.

        slack: float
            Delta slack parameter for composition of spends.

        Returns
        -------
        float
            Total delta spend.

        """
        delta_spend = [slack]
        for _, delta in spent_budget:
            delta_spend.append(delta)
        delta_spend.sort()

        # (1 - a) * (1 - b) = 1 - (a + b - a * b)
        prod = 0
        for delta in delta_spend:
            prod += delta - prod * delta

        return prod

    @staticmethod
    def load_default(accountant):
        """Loads the default privacy budget accountant if none is supplied, otherwise checks that the supplied
        accountant is a BudgetAccountant class.

        An accountant can be set as the default using the `set_default()` method.  If no default has been set, a default
        is created.

        Parameters
        ----------
        accountant : BudgetAccountant or None
            The supplied budget accountant.  If None, the default accountant is returned.

        Returns
        -------
        default : BudgetAccountant
            Returns a working BudgetAccountant, either the supplied `accountant` or the existing default.

        """
        if accountant is None:
            if BudgetAccountant._default is None:
                BudgetAccountant._default = BudgetAccountant()

            return BudgetAccountant._default

        if not isinstance(accountant, BudgetAccountant):
            raise TypeError(f"Accountant must be of type BudgetAccountant, got {type(accountant)}")

        return accountant

    def set_default(self):
        """Sets the current accountant to be the default when running functions and queries with diffprivlib.

        Returns
        -------
        self : BudgetAccountant

        """
        BudgetAccountant._default = self
        return self

    @staticmethod
    def pop_default():
        """Pops the default BudgetAccountant from the class and returns it to the user.

        Returns
        -------
        default : BudgetAccountant
            Returns the existing default BudgetAccountant.

        """
        default = BudgetAccountant._default
        BudgetAccountant._default = None
        return default
