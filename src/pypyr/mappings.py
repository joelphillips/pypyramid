'''
Created on Aug 20, 2010

@author: joel
'''
import numpy

def applyweights(wp, vp):    
    from numpy import newaxis
    wp = wp.reshape(wp.shape + (1,)*(3-len(wp.shape)))
    if len(vp.shape) == 2: vp = vp[:,:,newaxis]
    wvp = numpy.sum(wp[:,newaxis] * vp[:,:,newaxis,:], axis=3)
#    print "applyweights %s, %s, %s"%(wp.shape, vp.shape, wvp.shape)
    return wvp 

def psi(p):
    """ Map the reference pyramid to the infinite pyramid """
    
    zeta = p[:,2].reshape(-1,1)    
    pp = p / (1-zeta)
    for i in numpy.flatnonzero(zeta==1):
        if (numpy.abs(p[i,(0,1)]) < 1E-10).all(): pp[i] = numpy.array([0,0,numpy.inf]) # not perfect, as x,y <> 0 do not map to [0,0,inf] 
    return pp  

def psijac(p):
    xi = p[:,0].reshape(-1,1)
    eta = p[:,1].reshape(-1,1)
    zeta = p[:,2].reshape(-1,1)
    z1 = 1/(1-zeta)
    z2 = z1 * z1
    zeros = numpy.zeros_like(xi)
    return numpy.hstack([z1,zeros, xi*z2  ,zeros,z1,eta*z2,  zeros,zeros,z2 ] ).reshape(-1,3,3)

def psiinvjac(p):
    xi = p[:,0].reshape(-1,1)
    eta = p[:,1].reshape(-1,1)
    zeta = p[:,2].reshape(-1,1)
    z1 = (1-zeta)
    z2 = z1 * z1
    zeros = numpy.zeros_like(xi)
    return numpy.hstack([z1,zeros, -xi*z1  ,zeros,z1,-eta*z1,  zeros,zeros,z2 ] ).reshape(-1,3,3)
    

def psidet(p):
    zeta = p[:,2].reshape(-1,1)
    return 1/(1-zeta)**4
    

def derham3dweights(jac, invjac=None, dets=None):
    from numpy.linalg import tensorsolve, det
    from numpy import ones, transpose
    if invjac is None: invjac = lambda p : tensorsolve(jac(p), numpy.tile(numpy.eye(3), (len(p),1,1))) 
    if dets is None: dets = lambda p : map(det, jac(p))
    w0 = lambda p: ones(len(p))
    w1 = lambda p: transpose(jac(p), (0,2,1))
    w2 = lambda p: invjac(p) * dets(p).reshape(-1,1,1)
    w3 = dets
    return [w0,w1,w2,w3]    

def mapweights(map):
    return derham3dweights(map.jac, map.invjac, map.dets)

class Pullback(object):
    """ A callable object that returns the pullback H^*(f):M -> X of its argument, f: N -> X
    based on the (inverse) homeomorphism H: N -> M and a weight function w: N -> L(X,X), so that
    H^*(f)(p) = w(p) . f(g(p))"""  
    
    def __init__(self, map, weights=None):
        self.map = map
        self.weights = [lambda p: numpy.ones(len(p))] if weights is None else weights
    
    def __call__(self, f):
        def Hf(p):
            fHp = f(self.map(p))
            return self.apply(p, fHp)
        return Hf
    
    def apply(self, p, vals):
        """ Apply the weights to the underlying """
        return vals if self.weights is None else applyweights(self.weights[0](p), vals)
        
        
    def next(self):
        if len(self.weights) > 1:
            return Pullback(self.map, self.weights[1:])
    

def mapbasedpullback(map, s):
    """ Returns a callable that returns a pullback functional based on a map object (e.g. Affine)"""
    return Pullback(map.apply, mapweights(map)[s:])  

class Affine(object):
    def __init__(self, offset, linear):
        self.offset = offset
        self.linear = linear      
    
    def apply(self, p):
#        print "Affine apply", p.shape, self.linear.shape, self.offset.shape
        return (numpy.dot(p, self.linear.transpose()) + self.offset).reshape(-1,len(self.offset))
    
    def applyinv(self, q):        
        from numpy.linalg import solve
        return solve(self.linear, (q - self.offset.reshape(1,-1)).transpose()).transpose()
    
    def jac(self,p):
        return numpy.tile(self.linear, (len(p),1,1))
    
    def invjac(self, p):
        from numpy.linalg import inv
        return numpy.tile(inv(self.linear), (len(p),1,1))

    def dets(self,p):
        from numpy.linalg import det
        return numpy.ones(len(p)) * det(self.linear)
    
def buildaffine(pfrom, pto):
    from numpy.linalg import qr, solve
    from numpy import dot
    # treat the first point as the origin in each case
    # Taking transposes means that the first axis is the x,y,z component of the points in the second axis.
    if len(pfrom) > 1:
        F = (pfrom[1:] - pfrom[0]).transpose()
        T = (pto[1:] - pto[0]).transpose()
        
        # we want to find M such that M . F = T
        # if not enough points have been supplied, add in some orthogonal ones
        fmissing = F.shape[0] - F.shape[1]
        if fmissing:
            F = numpy.hstack((F, numpy.zeros((F.shape[0], fmissing))))
            T = numpy.hstack((T, numpy.zeros((T.shape[0], fmissing))))
            FQ = qr(F)[0]
            TQ = qr(T)[0]
            F[:,-fmissing:] = FQ[:,-fmissing:]
            T[:,-fmissing:] = TQ[:,-fmissing:]
        
        M = solve(F.transpose(),T.transpose())
        offset = pto[0] - dot(pfrom[0], M)
        return Affine(offset, M.transpose())
    
    
    
    
        
