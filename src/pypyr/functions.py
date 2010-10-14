'''
Created on Aug 20, 2010

@author: joel
'''
import numpy
from pypyr.timing import print_timing

def jacobid(n, a, b, d):
    """ (Derivatives of) Jacobi polynomials shifted to [0,1]"""
    from scipy.special.orthogonal import jacobi
    from math import factorial
    if d > n: return lambda x: numpy.zeros(x.shape)
    fac = factorial(a + b + n + d) / factorial(a + b + n)
    j = jacobi(n-d, a+d, b+d)
    return lambda x: fac * j(2 * x - 1)

#@print_timing
def jacobi2d(N,a,b,d,x):
    from scipy.special.orthogonal import poch
#    v = jacobi2(N-d, a+d, b+d,x)
    v = getJacobi(a+d,b+d)(N-d, x)   
    
#    facs = [factorial(a+b+d)/factorial(a+b)]
#    for n in range(1,N+1): facs.append(facs[-1]*(a+b+d+n) / (a+b+n))        
#    fv = numpy.array(facs)[d:,numpy.newaxis] * v
    fv = poch(numpy.arange(a+b+d+1, a+b+N+2), d)[:,numpy.newaxis] * v

    return numpy.vstack((numpy.zeros((d, len(x))), fv ))

def jacobi2(N,a,b,x):
    v = []
    x = 2*x - 1
    if N >= 0:
        v.append(numpy.ones(len(x)))
        if N >= 1:
            v.append(((a + b + 2) * x + (a-b))/2.0)
            for nn in range(2, N+1):
                n = nn * 1.0
                A=2*n*(n+a+b)/((2*n+a+b)*(2*n+a+b-1))
                B=-(a*a-b*b)/((2*n+a+b)*(2*n+a+b-2))
                C=2*(n+a-1)*(n+b-1)/((2*n+a+b-1)*(2*n+a+b-2))
                v.append((v[-1] * (x-B) - C*v[-2])/A)
#        print numpy.vstack(v).shape, N, len(x)
        return numpy.vstack(v).reshape(N+1,len(x))
    return numpy.zeros((0, len(x)))

jacobicache = {}
def getJacobi(a,b):
    j = jacobicache.get((a,b))
    if j is None:
        j = Jacobi(a,b)
        jacobicache[(a,b)] = j
    return j


class Jacobi(object):
    maxN = 20
#    maxrecent = 10
    def __init__(self, a,b):
        from collections import deque
        n = numpy.arange(0,self.maxN + 1, dtype=numpy.float_)
        self.A = 2*n*(n+a+b)/((2*n+a+b)*(2*n+a+b-1))
        self.B =-(a*a-b*b)/((2*n+a+b)*(2*n+a+b-2))
        self.C =2*(n+a-1)*(n+b-1)/((2*n+a+b-1)*(2*n+a+b-2))
        self.A[1] = 2.0/(a+b+2)
        self.B[1] = -(a-b)/(a+b+2.0)
        self.C[1] = 0
        
#        self.recentx = deque(maxlen = self.maxrecent)
#        self.recentvals = deque(maxlen = self.maxrecent)
        
    def __call__(self,N,x):
#        for rx,rv in zip(self.recentx, self.recentvals):
#            if numpy.array_equal(x,rx): 
##                print "cache hit"
#                return rv
#        v = self.eval(N,x)
#        self.recentx.append(x)
#        self.recentvals.append(v)
#        return v
#            
#        
#    def eval(self, N, x):    
        v = [numpy.zeros(len(x))]
        x = 2*x - 1
        if N >= 0:
            v.append(numpy.ones(len(x)))
            for n in range(1, N+1):
                v.append((v[-1] * (x-self.B[n]) - self.C[n]*v[-2])/self.A[n])
    #        print numpy.vstack(v).shape, N, len(x)
        return numpy.vstack(v)[1:,:]
        
        
    
class QSpace(object):
    def __init__(self, l,m,k, rmin=0, d = numpy.array([0,0,0])):
        if d[2] > 1: raise ValueError("Can only differentiate z once. d=%s"%d)
        self.l = l
        self.m = m
        self.k = k
        self.d = d
        self.rmin = rmin
        lm = min(l,m)
        self.nfns = (2*lm+3)*(lm+2) * (lm+1) / 6 + (lm+2)*(lm+1)*abs(l-m)/2 - self.rmin
        
#    @print_timing    
    def values(self,p):
        from math import factorial
        from numpy import newaxis
          
        x = p[:,0] 
        y = p[:,1] 
        z = p[:,2]
        (xd,yd,zd) = self.d 
        
        zeta = 1 / (1+z)
        zeta[z == numpy.inf] = 0
        vals = []
        
        pxk = jacobi2d(self.k,0,0,xd,x)
        pyk = jacobi2d(self.k,0,0,yd,y)
        for r in range(self.rmin, self.k+1):
#            pz = jacobid(r,0,2,0)(1-zeta) if zd == 0 else zeta**2 * jacobid(r,0,2,1)(1-zeta)
            dfac = (-1)**zd * factorial(r+zd-1)/factorial(r-1) if r > 0 else 1 if zd == 0 else 0
            pz = dfac * zeta**(r+zd)
#            px = [jacobid(n,0,0,xd)(x) for n in range(r+self.l+1-self.k)] # probably ought to try something other than legendre
#            py = [jacobid(n,0,0,yd)(y) for n in range(r+self.m+1-self.k)]
#            vals.extend([(pz * ppx * ppy).reshape(-1,1) for ppx in px for ppy in py])
#            
#            px = jacobi2d(r+self.l-self.k,0,0,xd,x)
#            py = jacobi2d(r+self.m-self.k,0,0,yd,y)

#            print px.shape, pxk[:max(0,r+self.l+1-self.k, :].shape, r, self.l, self.k
#            print r, self.l, self.m, self.k, self.d, (pxk[:max(0,r+self.l+1-self.k),newaxis, :] * pyk[newaxis,:max(0,r+1+self.m-self.k),:] * pz).shape, pz.shape
            vals.append((pxk[:max(0,r+self.l+1-self.k),newaxis, :] * pyk[newaxis,:max(0,r+1+self.m-self.k),:] * pz).reshape(-1, len(p)).transpose())
            
        V = numpy.hstack(vals)
        return V
    
    def powers(self):
        powers = []
        for r in range(self.k+1):
            powers.extend([[xp, yp, r] for xp in range(r+self.l+1-self.k) for yp in range(r+self.m+1-self.k)])
        return powers 
        
    
    def deriv(self, i):
        newd = self.d.copy()
        newd[i]+=1
        return QSpace(self.l, self.m, self.k, self.rmin, newd)    

class ZeroFns(object):
    def __init__(self, nfns):
        self.nfns = nfns
        
    def values(self, p):
        return numpy.zeros((len(p), self.nfns))
    
    def deriv(self, d):
        return self
    
class LinearComb(object):
    def __init__(self, fns, coeffs):
        self.fns = fns
        self.coeffs = coeffs
        self.nfns = fns[0].nfns
    
    def values(self, p):
        return numpy.sum(self.coeffs.reshape(-1,1,1) * numpy.array([fn.values(p) for fn in self.fns]), axis=0)
        
    def deriv(self, d):
        return LinearComb([f.deriv(d) for f in self.fns], self.coeffs)