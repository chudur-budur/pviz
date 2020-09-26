import unittest
import numpy as np
from numpy import testing as npt
from viz.utils import io

class TestIo(unittest.TestCase):
    
    def test_is_number(self):
        tests = ['123', '12.3', '1.2.3', '-123', '-1.2.3', '-1.-2.-3', \
                '12x3', '1e6', '1ex6', '1.2e6', '-1e6', '-1.1e6', '--1e6', \
                 '1.2-e6', '1.2e-6', '1.2e-06', '1-2e6', '3.4e3', '34e-06', \
                '3.4.2ee3', '1e-6', '1e+6', 'abc']
        y = [io.is_number(v) for v in tests]
        self.assertEqual(y, \
                [True, True, False, True, False, False, False, True, False, \
                True, True, True, False, False, True, True, False, True, True, 
                False, True, False, False])

    def test_cast(self):
        x = [float, int, str, None]
        y = [io.cast('5', dtype=v) for v in x]
        self.assertEqual(y, [5.0, 5, '5', 5])
    
if __name__ == '__main__':
    unittest.main()

