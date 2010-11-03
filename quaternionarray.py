"""
Quaternion array module

A Quaternion array is a ndarray object with 4 columns defined as x y z w
it can be created simply using np.array.

Example random quaternion array with 3 quaternions:
>>> np.random.random((3, 4))
array([[ 0.65587228,  0.53004948,  0.14372156,  0.67375345],
       [ 0.395546  ,  0.61416382,  0.53622605,  0.4199437 ],
       [ 0.93493976,  0.43419391,  0.24041596,  0.95863378]])
"""
from __future__ import division
import numpy as np

def array_dot(a, b):
    '''Dot product of a lists of arrays'''
    return np.sum(a*b, axis=1)

def inv(q):
    """Inverse of quaternion array q"""
    return q * np.array([-1,-1,-1,1])

def norm(q):
    """Normalize quaternion array q to unit quaternions"""
    return q/np.sqrt(array_dot(q,q))[:,np.newaxis]

def rotate(q, v):
    """Rotate vector or array of vectors v by quaternion q"""
    if v.ndim == 1:
        qv = np.append(v,0)
    else:
        qv = np.hstack([v,np.zeros((len(v),1))])
    return mult(mult(q,qv),inv(q))[:,:3]

def mult(p, q):
    """Multiply arrays of quaternions,
    see:
    http://en.wikipedia.org/wiki/Quaternions#Quaternions_and_the_geometry_of_R3
    """
    if p.ndim == 1 and q.ndim > 1:
        p = np.tile(p,(q.shape[0],1))
    if q.ndim == 1 and p.ndim > 1:
        q = np.tile(q,(p.shape[0],1))
    if q.ndim == 1 and p.ndim == 1:
        p = p.reshape((1,4))
        q = q.reshape((1,4))

    ps = p[:,3]
    qs = q[:,3]
    pv = p[:,:3]
    qv = q[:,:3]

    pq = np.empty_like(p)
    pq[:,3] =  ps * qs - array_dot(pv, qv)
    pq[:,:3] = ps[:,np.newaxis] * qv + pv * qs[:,np.newaxis] + np.cross(pv , qv)

    #opposite sign due to different convention on the basis vectors
    pq = -1 * pq
    return pq

def nlerp(targettime, time, q):
    """Nlerp, q quaternion array interpolated from time to targettime"""
    i_interp = np.interp(targettime, time, np.arange(len(time)))
    i_interp_int = np.floor(i_interp).astype(np.int)
    t_matrix = i_interp - i_interp_int
    #vertical array
    t_matrix = t_matrix[:,np.newaxis]
    q_interp = q[i_interp_int,:] * (1 - t_matrix) + q[np.clip(i_interp_int + 1,0,len(time)-1),:] * t_matrix
    return q_interp
