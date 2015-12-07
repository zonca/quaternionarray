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
#from IPython.Debugger import Tracer; debug_here = Tracer()

def arraylist_dot(a, b):
    '''Dot product of a lists of arrays, returns a column array'''
    if a.ndim == 1 and b.ndim == 1:
        return np.array([[np.dot(a,b)]])
    else:
        return np.sum(a*b, axis=1)[:,np.newaxis]

def inv(q):
    """Inverse of quaternion array q"""
    return q * np.array([-1,-1,-1,1])

def amplitude(v):
    return np.sqrt(arraylist_dot(v,v))

def norm(q):
    """Normalize quaternion array q or array list to unit quaternions"""
    normq = q/amplitude(q)
    if q.ndim == 1:
        normq = normq.flatten()
    return normq

def norm_inplace(q):
    """Normalize quaternion array q or array list to unit quaternions"""
    q /= amplitude(q)

def rotate(q, v):
    """Rotate vector or array of vectors v by quaternion q"""
    if v.ndim == 1:
        qv = np.append(v,0)
    else:
        qv = np.hstack([v,np.zeros((len(v),1))])
    out = mult(q,qv)
    out = mult(out, inv(q))
    return out[:,:3]


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
    pq[:,3] =  ps * qs 
    pq[:,3] -= arraylist_dot(pv, qv).flatten()
    pq[:,:3] = ps[:,np.newaxis] * qv 
    pq[:,:3] += pv * qs[:,np.newaxis] 
    pq[:,:3] += np.cross(pv , qv)

    #opposite sign due to different convention on the basis vectors
    #pq *= -1
    return pq

def nlerp(targettime, time, q):
    """Nlerp, q quaternion array interpolated from time to targettime"""
    i_interp_int, t_matrix = compute_t(targettime, time)
    q_interp = q[i_interp_int,:] * (1 - t_matrix) 
    q_interp += q[np.clip(i_interp_int + 1,0,len(time)-1),:] * t_matrix
    return norm(q_interp)

def slerp(targettime, time, q):
    """Slerp, q quaternion array interpolated from time to targettime"""
    #debug_here()
    i_interp_int, t_matrix = compute_t(targettime, time)
    q_interp = mult(q[np.clip(i_interp_int + 1,0,len(time)-1),:], inv(q[i_interp_int,:]))
    q_interp = pow(q_interp, t_matrix) 
    q_interp = mult(q_interp, q[i_interp_int,:])
    t_zero = (t_matrix == 0).flatten()
    q_interp[t_zero] = q[i_interp_int][t_zero]
    return q_interp

def compute_t(targettime, time):
    i_interp = np.interp(targettime, time, np.arange(len(time)))
    i_interp_int = np.floor(i_interp).astype(np.int)
    t_matrix = i_interp - i_interp_int
    #vertical array
    t_matrix = t_matrix[:,np.newaxis]
    return i_interp_int, t_matrix

def exp(q):
    """Exponential of a quaternion array"""
    normv = amplitude(q[:,:3])
    res = np.zeros_like(q)
    res[:,3:] = np.exp(q[:,3:]) * np.cos(normv)
    res[:,:3] = np.exp(q[:,3:]) * q[:,:3] / normv 
    res[:,:3] *= np.sin(normv)
    return res

def ln(q):
    """Natural logarithm of a quaternion array"""
    normq = amplitude(q)
    res = np.zeros_like(q)
    res[:,3:] = np.log(normq)
    res[:,:3] = norm(q[:,:3])
    res[:,:3] *= np.arccos(q[:,3:]/normq)
    return res

def pow(q, p):
    """Real power of a quaternion array"""
    return exp(ln(q)*p)
    
def rotation(axis, angle):
    """Rotation quaternions of angles [rad] around axes [already normalized]"""
    try:
        angle = angle[:,None]
    except:
        pass
    return np.hstack([np.asarray(axis)*np.sin(angle/2.),np.cos(angle/2.)])

def to_axisangle(q):
    angle = 2 * np.arccos(q[3])
    if angle == 0:
        axis = np.zeros(3)
    else:
        axis = q[:3] / np.sin(angle/2.)
    return axis, angle

def to_rotmat(q):
    """Rotation matrix"""
    s = q[3]
    v = q[:3]
    return (s**2 -  np.dot(v,v)) * np.eye(3) + \
            2 * v[:,None] * v  + \
            2 * s * np.array([[0    , -q[2], q[1] ],
                              [q[2] , 0    , -q[0] ],
                              [-q[1], q[0] , 0     ]])
                                

def from_rotmat(rotmat):
    rotmat = np.asarray(rotmat)
    r = np.sqrt(1 + rotmat[0,0] - rotmat[1,1] - rotmat[2,2])
    return np.array([
        r,
        (rotmat[0,1] + rotmat[1,0])/r,
        (rotmat[0,2] + rotmat[2,0])/r,
        (rotmat[2,1] - rotmat[1,2]) / r
        ])/2.

def from_vectors(v1, v2):
    v = np.cross(v1, v2)
    s = np.sqrt(np.linalg.norm(v1)**2 * np.linalg.norm(v2)**2) + np.dot(v1, v2)
    return norm(np.concatenate([v, [s]]))
