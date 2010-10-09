'''
Created on Aug 20, 2010

@author: joel
'''
import unittest
import numpy
from pypyr.functions import *

class TestJacobi(unittest.TestCase):
    def testJacobi2(self):
        for a in range(3):
            for b in range(3):
                for d in range(4):
                    N = 10
                    x = numpy.linspace(0,1,10)
                    j1 = numpy.vstack([jacobid(n,a,b,d)(x) for n in range(N+1)])
                    j2 = jacobi2d(N,a,b,d,x)
                    numpy.testing.assert_array_almost_equal(j1,j2)
    
    def testJacobi2and3(self):
        from scipy.special.orthogonal import jacobi
        N = 5
        x = numpy.linspace(0,1,6)
        for a in range(4):
            for b in range(4):
                j1 = numpy.vstack([jacobi(n,a,b)(2*x-1) for n in range(N+1)])
                j2 = jacobi2(N,a,b,x)
                j3 = Jacobi(a,b)(N,x)
                numpy.testing.assert_array_almost_equal(j1,j3)
                numpy.testing.assert_array_almost_equal(j2,j3)
        

class TestQSpace(unittest.TestCase):
    
    def testSize(self):
        for c in [[2,2,2],[2,1,2],[3,3,3],[2,2,3]]:  # l,m,k
            q = QSpace(*c)
            p = numpy.array([[1,0,0],[0.5,0.5,0],[0.5,0.5,1], [0.2,0.2,3]]) 
            qp = q.values(p)
            self.assertEqual(qp.shape[1], q.nfns)
        
        
    def testDeriv(self):
        import numpy.testing
        q = QSpace(2,2,2)
        
        qdx = q.deriv(0)
        qdy = q.deriv(1)
        qdz = q.deriv(2)
        
        p = numpy.array([[1,0,0],[0.5,0.5,0],[0.5,0.5,1], [0.2,0.2,3]]) 
        h = 1.0E-10
        phx = p + [h,0,0]
        qdxp = qdx.values(p)
        qdxph = (q.values(phx) - q.values(p)) / h
        phy = p + [0,h,0]
        qdyp = qdy.values(p)
        qdyph = (q.values(phy) - q.values(p)) / h
        phz = p + [0,0,h]
        qdzp = qdz.values(p)
        qdzph = (q.values(phz) - q.values(p)) / h
                
        numpy.testing.assert_array_almost_equal(qdxp, qdxph, 5)
        numpy.testing.assert_array_almost_equal(qdyp, qdyph, 5)
        numpy.testing.assert_array_almost_equal(qdzp, qdzph, 5)
        
