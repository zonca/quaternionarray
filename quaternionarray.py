from __future__ import division
import numpy as np

def inv(q):
    """Inverse of quaternion array q"""
    return q * np.array([-1,-1,-1,1])

def norm(q):
    """Normalize quaternion array q to unit quaternions"""
    return q/np.sqrt(np.sum(np.square(q),axis=1))[:,np.newaxis]

def rotate(q, v):
    """Rotate or array of vectors v by quaternion q"""
    if v.ndim == 1:
        qv = np.append(v,0)
    else:
        qv = np.hstack([v,np.zeros((len(v),1))])
    return mult(mult(q,qv),inv(q))[:,:3]

def mult(p, q):
    '''Multiply arrays of quaternions, ndarray objects with 4 columns defined as x y z w
    see:
    http://en.wikipedia.org/wiki/Quaternions#Quaternions_and_the_geometry_of_R3
    '''
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
    pq[:,3] =  ps * qs - np.sum(pv*qv, axis=1)
    pq[:,:3] = ps[:,np.newaxis] * qv + pv * qs[:,np.newaxis] + np.cross(pv , qv)

    #opposite sign due to different convention on the basis vectors
    pq = -1 * pq
    return pq

def nlerp(targettime, time, q):
    '''Nlerp, q quaternion array interpolated from time to targettime'''
    i_interp = np.interp(targettime, time, np.arange(len(time)))
    i_interp_int = np.floor(i_interp).astype(np.int)
    t_matrix = i_interp - i_interp_int
    #vertical array
    t_matrix = t_matrix[:,np.newaxis]
    q_interp = q[i_interp_int,:] * (1 - t_matrix) + q[i_interp_int + 1,:] * t_matrix
    return q_interp
