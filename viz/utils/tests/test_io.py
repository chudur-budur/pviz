import unittest
import numpy as np
from numpy import testing as npt
from viz.utils import io

class TestIo(unittest.TestCase):
    
    def test_is_number(self):
        x = ['123', '123.4', '12.3.4', '-123.4', '123e10']
        y = [io.is_number(v) for v in x]
        self.assertEqual(y, [True, True, False, True, False])

    def test_cast(self):
        x = [float, int, str, None]
        y = [io.cast('5', dtype=v) for v in x]
        self.assertEqual(y, [5.0, 5, '5', 5])
    
if __name__ == '__main__':
    unittest.main()

