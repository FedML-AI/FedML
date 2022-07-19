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
Transform wrappers for differential privacy mechanisms to extend their use to alternative data types.

Notes
-----
The naming convention for new transforms is to describe the `pre-transform` action, i.e. the action performed on the
data to be ingested by the mechanism.  For transforms without a `pre-transform`, the `post-transform` action should be
described.

"""
from fedml.core.differential_privacy.mechanisms.transforms.base import DPTransformer

from fedml.core.differential_privacy.mechanisms.transforms.roundedinteger import RoundedInteger
from fedml.core.differential_privacy.mechanisms.transforms.stringtoint import StringToInt
from fedml.core.differential_privacy.mechanisms.transforms.inttostring import IntToString
