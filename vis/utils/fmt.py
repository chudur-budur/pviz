"""`fmt.py` -- A Collection of Different Utility Functions for Pretty Formatting
    
    This module provide different utility functions for pretty printing of different
    collections objects. Like lists, dict, etc. 

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

__all__ = ["fmt"]

def strv(v):
    r""" Stringize a singleton.

    If the singleton `v` is float or int, format into float 
    with 2 decimal places. Otherwise just return the string.

    Parameters
    ----------
    `v` : object
        The input singleton data type.

    Returns
    -------
    `s` : str
        The output string.
    """
    s = ""
    if type(v) == float or type(v) == int:
        s = "{:.2f}".format(v)
    elif type(v) == str:
        s = v
    return s

def fmt(obj):
    r""" Formats an object `obj` (non-recursively)

    A very simple formatter for object `obj`. Does not do anything
    recursively.

    Parameters
    ----------
    'obj' : object
        The input object.

    Returns
    -------
    `s` : str
        The output string.

    Todo
    ----
    Make this module recursive so that it can work with any object.

    """
    s = ""
    if isinstance(obj, list):
        s = s + "["
        for item in obj:
            if isinstance(item, list):
                s = s + "[" + ", ".join([strv(v) for v in item]) + "]\n "
            else:
                s = s + strv(item) + ", "
        s = s[:-2] + "]"
    return s

