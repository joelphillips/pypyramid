'''
Created on Aug 21, 2010

@author: joel
'''
import unittest

from pypyr.mesh import *

class MockMesh(object):
        def __init__(self):
            self.points = {}
            self.pyramids = []
            self.boundary = {}
        def addPoint(self,id, p):
#            print "addPoint %s"%p
            self.points[id] = p
        def addPyramid(self,ids):
#            print "addPyramid %s"%ids 
            self.pyramids.append(ids)
        def addBoundary(self, tag, ids):
            self.boundary[tag] = ids

class TestMesh(unittest.TestCase):    
    def testBuildMesh(self):
        for n in range(1,6):
            mockmesh = MockMesh()
            Tag = "Boundary%s"%n
            buildcubemesh(n,mockmesh, Tag)
            self.assertEqual(len(mockmesh.points), n**3 + (n+1)**3)
            self.assertEqual(len(mockmesh.pyramids), 6 * n**3)
            self.assertEqual(len(mockmesh.boundary[Tag]), (n+1)**3 - (n-1)**3)
            
class TestElementFinder(unittest.TestCase):
    def testElementFinder(self):
        n = 2
        np = 10
        ef = ElementFinder()
        buildcubemesh(n, ef)
        p  = numpy.linspace(0,1,np)[numpy.mgrid[0:np,0:np,0:np].reshape(3,-1)].transpose()
        etop = ef.elementPointMap(p)
        self.assertEqual(sum(map(len, etop)), np**3)
        
class TestBoundaryQuadrature(unittest.TestCase):
    def testBoundaryQuad(self):
        from pypyr.utils import squarequadrature, trianglequadrature
        tag = "TAG"
        N = 1
        n = 3
        bq = BoundaryQuadrature()
        buildcubemesh(N, bq, boundarytag = tag)
        s = 0
        sx = 0
        for x,w, ns in bq.getQuadratures(tag, squarequadrature(n), trianglequadrature(n)):
            if len(x):
                s+=sum(w)
                sx += numpy.dot(x[:,0], w)
                x0 = x[0]
#                print ns, len(x), len(w)
                for xp,np in zip(x, ns):
                     self.assertEqual(numpy.dot(xp - x0, np), 0)
                     self.assertTrue(numpy.dot(xp - numpy.array([0.5,0.5,0.5]), np) > 0)
                    
#                     print xp, np, numpy.dot(xp - numpy.array([0.5,0.5,0.5]), np)
                     
        self.assertAlmostEqual(s, 6)
        self.assertAlmostEqual(sx, 3)