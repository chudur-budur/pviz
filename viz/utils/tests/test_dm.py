import unittest
import numpy as np
from numpy import testing as npt
from viz.utils import dm

class TestDm(unittest.TestCase):
    A = np.diagflat([[1,2,3], [4,5,6]]).astype(int)
    B = np.arange(0,6,1).astype(int)
    x, y = np.mgrid[0:5, 5:10]
    C = np.c_[x.ravel(), y.ravel()]

    def test_nadir(self):
        y = dm.nadir(self.A)
        npt.assert_equal(y, np.array([1, 2, 3, 4, 5, 6]))
    
    def test_ideal(self):
        y = dm.ideal(self.A)
        npt.assert_equal(y, np.array([0, 0, 0, 0, 0, 0]))

    def test_knees(self):
        y = dm.knees(self.B)
        npt.assert_equal(y, np.array([5]))

    def test_tradeoff(self):
        y = dm.tradeoff(self.C)
        npt.assert_equal(y[0], np.array([np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]))
        npt.assert_equal(y[1], np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23]))

if __name__ == '__main__':
    unittest.main()
