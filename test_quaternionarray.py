import numpy as np
import unittest

import sys
import os
sys.path.append(os.getcwd())
import quaternionarray as qarray

class TestQuaternionArray(unittest.TestCase):
    

    def setUp(self):
        #data
        self.q1 = np.array([ 0.50487417,  0.61426059,  0.60118994,  0.07972857])
        self.q1inv = np.array([ -0.50487417,  -0.61426059,  -0.60118994,  0.07972857])
        self.q2 = np.array([ 0.43561544,  0.33647027,  0.40417115,  0.73052901])
        self.qtonormalize = np.array([[1,2,3,4],[2,3,4,5]])
        self.qnormalized = np.array([[0.18257419,  0.36514837,  0.54772256,  0.73029674],[ 0.27216553,  0.40824829,  0.54433105,  0.68041382]])
        self.vec = np.array([ 0.57734543,  0.30271255,  0.75831218])
        self.vec2 = np.array([[ 0.57734543,  8.30271255,  5.75831218],[ 1.57734543,  3.30271255,  0.75831218]])
        #results from Quaternion
        self.mult_result = np.array([-0.44954009, -0.53339352, -0.37370443,  0.61135101])
        self.rot_by_q1 = np.array([0.4176698, 0.84203849, 0.34135482])
        self.rot_by_q2 = np.array([0.8077876, 0.3227185, 0.49328689])

        # error on floating point equality tests
        self.EPSILON = 1e-7

    def test_arraylist_dot_onedimarrays(self):
        np.testing.assert_array_almost_equal(qarray.arraylist_dot(self.vec, self.vec +1), np.dot(self.vec, self.vec +1)) 

    def test_arraylist_dot_1dimbymultidim(self):
        np.testing.assert_array_almost_equal(qarray.arraylist_dot(self.vec2, self.vec), np.dot(self.vec2,self.vec)) 

    def test_arraylist_dot_multidim(self):
        result = np.hstack(np.dot(v1,v2) for v1,v2 in zip(self.vec2, self.vec2+1))
        np.testing.assert_array_almost_equal(qarray.arraylist_dot(self.vec2, self.vec2 +1), result) 

    def test_inv(self):
        self.assertTrue((qarray.inv(self.q1) - self.q1inv).std() < self.EPSILON)

    def test_norm(self):
        np.testing.assert_array_almost_equal(qarray.norm(self.q1), self.q1/np.linalg.norm(self.q1))
        self.assertTrue((qarray.norm(self.qtonormalize) - self.qnormalized).std() < self.EPSILON)


    def test_mult_onequaternion(self):
        my_mult_result = qarray.mult(self.q1, self.q2)
        self.assertEquals( my_mult_result.shape[0], 1)
        self.assertEquals( my_mult_result.shape[1], 4)
        self.assertTrue((my_mult_result - self.mult_result).std() < self.EPSILON)

    def test_mult_qarray(self):
        dim = (3, 1)
        qarray1 = np.tile(self.q1, dim)
        qarray2 = np.tile(self.q2, dim)
        my_mult_result = qarray.mult(qarray1, qarray2)
        self.assertTrue((my_mult_result - np.tile(self.mult_result,dim)).std() < self.EPSILON)

    def test_rotate_onequaternion(self):
        my_rot_result = qarray.rotate(self.q1, self.vec)
        self.assertTrue((my_rot_result - self.rot_by_q1).std() < self.EPSILON)
        
    def test_rotate_qarray(self):
        my_rot_result = qarray.rotate(np.vstack([self.q1,self.q2]), self.vec)
        self.assertTrue((my_rot_result - np.vstack([self.rot_by_q1, self.rot_by_q2])).std() < self.EPSILON)

if __name__ == '__main__':
    # better to use nose
    unittest.main()
