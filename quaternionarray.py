import numpy as np

def rotate(q, v):
    """Rotate or array of vectors v by quaternion q"""
    if v.ndim == 1:
        qv = np.append(v,0)
    else:
        qv = np.hstack([v,np.zeros((len(v),1))])
    return mult(mult(q,qv),q * np.array([1,1,1,-1]))[:,:3]

def mult(p, q):
    '''Multiply arrays of quaternions, ndarray objects with 4 columns defined as x y z w
    see http://en.wikipedia.org/wiki/Quaternions#Quaternions_and_the_geometry_of_R3
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
    pq[:,3] = ps * qs - np.sum(pv*qv, axis=1)
    pq[:,:3] = ps[:,np.newaxis] * qv + pv * qs[:,np.newaxis] + np.cross(pv , qv)

    return pq
