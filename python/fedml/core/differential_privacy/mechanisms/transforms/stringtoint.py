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
StringToInt DP transformer, for using string-valued data with integer-valued mechanisms.
"""
from fedml.core.differential_privacy.mechanisms.transforms.base import DPTransformer


class StringToInt(DPTransformer):
    """
    StringToInt DP transformer, for using string-valued data with integer-valued mechanisms.

    Useful when using ordered, string-valued data with :class:`.Geometric`.
    """

    def pre_transform(self, value):
        """Transforms the input to be integer-valued for ingestion by the mechanism.

        Parameters
        ----------
        value : float or string
            Input value to be transformed.

        Returns
        -------
        int
            Transformed input value

        """
        return int(value)

    def post_transform(self, value):
        """Transforms the output of the mechanism to be string-valued.

        Parameters
        ----------
        value : float or string
            Mechanism output to be transformed.

        Returns
        -------
        string
            Transformed output value.

        """
        return str(value)
