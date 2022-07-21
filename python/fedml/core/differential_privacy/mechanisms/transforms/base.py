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
Core utilities for DP transformers.
"""
from fedml.core.differential_privacy.mechanisms.base import DPMachine


class DPTransformer(DPMachine):
    """
    Base class for DP transformers.  DP Transformers are simple wrappers for DP Mechanisms to allow mechanisms to be
    used with data types and structures outside their scope.

    A :class:`.DPTransformer` must be initiated with a :class:`.DPMachine` (either another :class:`.DPTransformer`, or a
    :class:`.DPMechanism`).  This allows many instances of :class:`.DPTransformer` to be chained together, but the chain
    must terminate with a :class:`.DPMechanism`.

    """
    def __init__(self, parent):
        if not isinstance(parent, DPMachine):
            raise TypeError("Data transformer must take a DPMachine as input")

        self.parent = parent

    def pre_transform(self, value):
        """Performs no transformation on the input data, and is ingested by the mechanism as-is.

        Parameters
        ----------
        value : float or string
            Input value to be transformed.

        Returns
        -------
        float or string
            Transformed input value
        """
        return value

    def post_transform(self, value):
        """Performs no transformation on the output of the mechanism, and is returned as-is.

        Parameters
        ----------
        value : float or string
            Mechanism output to be transformed.

        Returns
        -------
        float or string
            Transformed output value.

        """
        return value

    def randomise(self, value):
        """
        Randomise the given value using the :class:`.DPMachine`.

        Parameters
        ----------
        value : float or string
            Value to be randomised.

        Returns
        -------
        float or string
            Randomised value, same type as `value`.

        """
        transformed_value = self.pre_transform(value)
        noisy_value = self.parent.randomise(transformed_value)
        output_value = self.post_transform(noisy_value)
        return output_value
