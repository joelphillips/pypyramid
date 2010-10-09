'''
Created on Aug 20, 2010

@author: joel
'''
import unittest
import numpy

from pypyr.utils import *
from pypyr.shapefunctions import *
       
class TestRForms(unittest.TestCase):
    def testR0Form(self):
        """ Test that we can interpolate all expected polynomials for the 0-forms"""
        from numpy.linalg import solve, det
        maxk = 7
        for k in range(1,maxk):
            rforms = buildRForms(k)
            R0 = rforms[0]
            p = pyrpoints(k)
            p95 = pyrpoints(k)*0.95
            p2 = pyrpoints(k*2)
            V0p = R0.values(p)
            V0p2 = R0.values(p2).squeeze()
            
#            print k, det(V0p.squeeze())
            
            V0Dp95 = R0.D().values(p95)
            
            
            for coeffs in [numpy.array([a,b,c]) for a in range(k+1) for b in range(k+1-a) for c in range(k+1-a-b)]:        
                polyp = polyn(coeffs, p)
                X0 = solve(V0p.squeeze(), polyp)
                polyp2 = polyn(coeffs, p2)
                numpy.testing.assert_array_almost_equal(numpy.dot(V0p2, X0), polyp2)
                
                # whilst we're at it, lets check the gradients too:
                if (coeffs > 1).all():
                    VDX = numpy.tensordot(V0Dp95, X0, ([1], [0]))
                    Dpolyp = numpy.hstack([coeffs[j] * polyn(coeffs - numpy.eye(3)[j],p95)[:,numpy.newaxis] for j in [0,1,2]])
#                    print Dpoly(p95), VDX
                    numpy.testing.assert_array_almost_equal(VDX, Dpolyp)
                
                
    def testR12Form(self):
        from numpy.linalg import lstsq
        maxk = 5
        for k in range(1,maxk):
            rforms = buildRForms(k)
            for R in rforms[1:3]:
                p = pyrpoints(k)*0.95
                p2 = pyrpoints(k*2)*0.95
                V1p = R.values(p)
                V1p2 = R.values(p2)
                V1pR = numpy.transpose(V1p, (2,0,1)).reshape(len(p)*3, R.nfns)
                V1p2R = numpy.transpose(V1p2, (2,0,1)).reshape(len(p2)*3, R.nfns)                        
                
                for coeffs in [(a,b,c) for a in range(k) for b in range(k-a) for c in range(k-a-b)]:        
                    poly = lambda p: numpy.product([p[:,i]**coeffs[i] for i in [0,1,2]], axis = 0)
                    for r in numpy.eye(3):
                        polyp = poly(p)
                        polypR = numpy.outer(r,polyp).flatten()
                        X1 = lstsq(V1pR, polypR)[0]
                        polyp2 = poly(p2)
                        polypR2 = numpy.outer(r,polyp2).flatten()
                        numpy.testing.assert_array_almost_equal(numpy.dot(V1p2R, X1), polypR2)      
