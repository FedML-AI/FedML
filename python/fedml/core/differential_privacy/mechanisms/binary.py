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
The binary mechanism for differential privacy.

"""
import numpy as np

from fedml.core.differential_privacy.mechanisms.base import DPMechanism
from fedml.core.differential_privacy.utils import copy_docstring


class Binary(DPMechanism):
    r"""The classic binary mechanism in differential privacy.

    Given a binary input value, the mechanism randomly decides to flip to the other binary value or not, in order to
    satisfy differential privacy.

    Paper link: https://arxiv.org/pdf/1612.05568.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    value0 : str
        0th binary label.

    value1 : str
        1st binary label.  Cannot be the same as ``value0``.

    Notes
    -----
    * The binary attributes, known as `labels`, must be specified as strings.  If non-string labels are required (e.g.
      integer-valued labels), a :class:`.DPTransformer` can be used (e.g. :class:`.IntToString`).

    """
    def __init__(self, *, epsilon, value0, value1):
        super().__init__(epsilon=epsilon, delta=0.0)
        self.value0, self.value1 = self._check_labels(value0, value1)

    @classmethod
    def _check_labels(cls, value0, value1):
        if not isinstance(value0, str) or not isinstance(value1, str):
            raise TypeError("Binary labels must be strings. Use a DPTransformer  (e.g. transformers.IntToString) for "
                            "non-string labels")

        if len(value0) * len(value1) == 0:
            raise ValueError("Binary labels must be non-empty strings")

        if value0 == value1:
            raise ValueError("Binary labels must not match")

        return value0, value1

    def _check_all(self, value):
        super()._check_all(value)
        self._check_labels(self.value0, self.value1)

        if not isinstance(value, str):
            raise TypeError("Value to be randomised must be a string")

        if value not in [self.value0, self.value1]:
            raise ValueError(f"Value to be randomised is not in the domain {{\"{self.value0}\", \"{self.value1}\"}}, "
                             f"got \"{value}\".")

        return True

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : str
            The value to be randomised.

        Returns
        -------
        str
            The randomised value.

        """
        self._check_all(value)

        indicator = 0 if value == self.value0 else 1

        unif_rv = self._rng.random() * (np.exp(self.epsilon) + 1)

        if unif_rv > np.exp(self.epsilon) + self.delta:
            indicator = 1 - indicator

        return self.value1 if indicator else self.value0
