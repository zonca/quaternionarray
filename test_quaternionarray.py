import numpy as np
import unittest

import sys
import os
sys.path.append(os.getcwd())
import quaternionarray as qarray

class TestQuaternionArray(unittest.TestCase):
    

    def setUp(self):
        self.q1 = np.array([ 0.50487417,  0.61426059,  0.60118994,  0.07972857])
        self.q2 = np.array([ 0.43561544,  0.33647027,  0.40417115,  0.73052901])
        self.vec = np.array([ 0.57734543,  0.30271255,  0.75831218])
        self.mult_result = np.array([-0.44954009, -0.53339352, -0.37370443,  0.61135101])
        # error on floating point equality tests
        self.EPSILON = 1e-7

    def test_mult_onearray(self):
        my_mult_result = -1 *qarray.mult(self.q1,self.q2)
        print(my_mult_result)
        print (my_mult_result - self.mult_result).std() 
        assert (my_mult_result - self.mult_result).std() < self.EPSILON

        
        

    
