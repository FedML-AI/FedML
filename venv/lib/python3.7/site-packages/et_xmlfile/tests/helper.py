from __future__ import absolute_import
# Copyright (c) 2010-2015 openpyxl

# Python stdlib imports
from lxml.doctestcompare import LXMLOutputChecker, PARSE_XML


def compare_xml(generated, expected):
    """Use doctest checking from lxml for comparing XML trees. Returns diff if the two are not the same"""
    checker = LXMLOutputChecker()

    class DummyDocTest():
        pass

    ob = DummyDocTest()
    ob.want = expected

    check = checker.check_output(expected, generated, PARSE_XML)
    if check is False:
        diff = checker.output_difference(ob, generated, PARSE_XML)
        return diff
