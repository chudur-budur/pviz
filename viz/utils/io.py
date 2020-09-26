"""io.py -- Different utility functions for I/O

    This module provides different utility functions for I/O.

    Copyright (C) 2016
    Computational Optimization and Innovation (COIN) Laboratory
    Department of Computer Science and Engineering
    Michigan State University
    428 S. Shaw Lane, Engineering Building
    East Lansing, MI 48824-1226, USA

.. moduleauthor:: AKM Khaled Talukder <talukde1@msu.edu>

"""

import os
import ast
import numpy as np

__all__ = ["loadtxt", "savetxt", "is_number"]

def is_num(s):
    r""" Inner function for `is_number()`
    """
    return s[1:].replace('.','',1).isdigit() \
            if s[0] == '-' else s.replace('.','',1).isdigit()
        
def is_number(s):
    r""" Returns True is string is a number. 
    
    This is a better solution than all the approaches described here:
    https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float  
    
    Parameters
    ----------
    s : str
        A string representing a number, like '-123', '12.3', '12e-06' etc.
    
    Returns
    -------
    bool : bool
    """
    if is_num(s):
        return True
    else:
        p = s.split('e')
        if len(p) != 2:
            return False
        else:
            return (is_num(p[0]) and is_num(p[1]))

def cast(x, dtype):
    r"""Typecasting a scalar value `x`.

    A very crude implementation of typecasting 
    a scalar `x` with respect to `dtype`.

    Parameters
    ----------
    x : str
        A scalar variable `x`
    dtype : data-type
        A data-type literal, can be `int`, `float`, `str` etc.

    Returns
    -------
    dtype(x) : typecasted `x`
    
    """
    if dtype is float:
        return float(x)
    elif dtype is int:
        return int(x)
    elif dtype is str:
        return str(x)
    else:
        # fall back
        return ast.literal_eval(x)

def tolist(A):
    r""" Recursively apply `tolist()` on an numpy array
    
    This fucnction recursively applies `numpy.tolist()` on an numpy array.
    It will work for any array, even on a jagged one.

    Parameters
    ----------
    A : An ndarray or a list or  alist of ndarrays or ndarrays of lists
        The input `ndarray` or `list` to be converted.

    Returns
    -------
    list : A list or list of lists
        Returns a fully converted list.
    """
    if isinstance(A, np.ndarray):
        return tolist(A.tolist())
    elif isinstance(A, list):
        return [tolist(a) for a in A]
    elif isinstance(A, tuple):
        return tuple(tolist(a) for a in A)
    else:
        return A

def loadtxt(fname, dtype=float, delimiter=None):
    r"""Load an array from a file, very similar to `npyio.loadtxt()`.

    Since `npyio.loadtxt()` is not useful when we want to load a jagged array
    from a text file, we have added this function exactly for that reason.    

    Parameters
    ----------
    fname : str or pathlib.Path
        A filename or a file path in string or `pathlib.Path` to read.
    dtype : data-type, optional
        Data-type of the resulting array. It can be `int`, `float`, `str` etc.
        See `cast()` function for details. Default `float` when optional. 
    delimiter : str, optional
        The string used to separate values. Default is whitespace when optional.

    Returns
    -------
    X : ndarray
        Data read from the text file.
    """
     
    if os.path.exists(fname):
        try:
            fp = open(fname, 'r')
            X = []
            for l in fp:
                a = [cast(v.strip(), dtype) for v in l.strip().split(delimiter)]
                X.append(np.array(a))
            fp.close()
            return np.array(X, dtype=object)
        except IOError:
            print("Coudn't open file {:s}".format(fname))
    else:
        raise OSError("File {0:s} not found.".format(fname))

def savetxt(fname, X, fmt='{:.18e}', delimiter=' ', newline='\n'):
    r"""Save an array to a file, very similar to `npyio.savetxt()`.

    Since `npyio.savetxt()` is not useful when we want to save a jagged array
    in a text format, we have added this function exactly for that reason.

    Parameters
    ----------
    fname : str
        A file path in string.
    X : 1D or 2D array_like
        Data to be saved to a text file. It can be a jagged array.
    fmt : str or sequence of str's, optional
        Python3 formatting string, i.e. '{:d}' for `int`, `{:f}` for `float` etc.
        Default '{:.18e}' when optional.
    delimiter : str, optional
        String or character separating columns. A whitespace when default.
    newline : str, optional
        String or character separating lines. Default `\n` when optional.
    """
    try:
        fp = open(fname, 'w+')
        for i in range(X.shape[0]-1):
            fp.write(delimiter.join([fmt.format(v) for v in X[i]]) + newline)
        fp.write(delimiter.join([fmt.format(v) for v in X[X.shape[0]-1]]))
        fp.close()
    except IOError:
        print("Coudn't open file {:s}".format(fname))
