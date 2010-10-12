'''
Created on Aug 21, 2010

@author: joel
'''
import unittest
from pypyr.shapefunctions import *
from pypyr.assembly import *
from pypyr.elements import *
from pypyr.mappings import *

from pypyr.utils import pyrpoints

import numpy as np

class TestDegrees(unittest.TestCase):
    
    def testRefPoints(self):
        for k in range(2, 10):
            self.assertEqual(len(edgepoints(k)), k)
            self.assertEqual(len(trianglepoints(k)), k*(k+1)/2)
            self.assertEqual(len(squarepoints(k,k)), k*k)
            
    def testh1pyramid(self):
        """ Initialise an h1 pyramid and test that its shape functions are dual to the external degrees """
        np.set_printoptions(precision = 4, linewidth=200)
        for k in range(1,6):
            degrees = H1Elements(k)
            points = numpy.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[1,1,1]])
            cs = []
            for p in points: cs.append(degrees.vertex(p[np.newaxis,:]))
            for e in [[0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4]]: cs.append(degrees.edge(points[e]))
            for t in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]: cs.append(degrees.triangle(points[t]))
            cs.append(degrees.quad(points[[0,3,1]]))
            elt = degrees.pyramid(points, cs)
            
#            map = buildaffine(points[[0,1,3,4]], points[[0,1,3,4]])            
#            elt = Element(rform, cs, mapbasedpullback(map))
    
            np.set_printoptions(precision = 4, suppress=True, linewidth=200)
            dofvals = numpy.concatenate([c.evaluatedofs(elt.values) for c in cs if c is not None], axis=0)
            ndof, nfn = dofvals.shape
            numpy.testing.assert_array_almost_equal(dofvals, numpy.hstack((numpy.eye(ndof), numpy.zeros((ndof,nfn - ndof )))))
    
    def testDegreeSet(self):
        for k in range(1,6):
            degrees = H1Elements(k)
            points = numpy.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[1,1,1]])
            cs = []
            for p in points: cs.append(degrees.vertex(p[np.newaxis,:]))
            for e in [[0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4]]: cs.append(degrees.edge(points[e]))
            for t in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]: cs.append(degrees.triangle(points[t]))
            cs.append(degrees.quad(points[[0,3,1]]))
            g = lambda p: p[...,numpy.newaxis]
            ds = DegreeSet(cs)
            numpy.testing.assert_array_almost_equal(ds.evaluatedofs(g), numpy.vstack([c.evaluatedofs(g) for c in cs if c is not None]))        
        
    def testhdivpyramid(self):
        for k in range(1,6):
            degrees = HdivElements(k)
            points = numpy.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[1,1,1]])
            cs = []
            for t in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]: cs.append(degrees.triangle(points[t]))
            cs.append(degrees.quad(points[[0,3,1]]))
            elt = degrees.pyramid(points, cs)
            dofvals = numpy.concatenate([c.evaluatedofs(elt.values) for c in cs if c is not None], axis=0)
            ndof, nfn = dofvals.shape
            numpy.testing.assert_array_almost_equal(dofvals, numpy.hstack((numpy.eye(ndof), numpy.zeros((ndof,nfn - ndof )))))
            
#    def testhcurlpyramid(self):
#        for k in range(1,6):
#            print "H(curl) ", k
#            degrees = HcurlElements(k)
#            points = numpy.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[1,1,1]])
#            cs = []
#            for e in [[0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4]]: cs.append(degrees.edge(points[e]))            
#            for t in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]: cs.append(degrees.triangle(points[t]))
#            cs.append(degrees.quad(points[[0,3,1]]))
##            for c in cs:
##                print c.points.shape, c.pullback.map.linear.transpose().shape
#            elt = degrees.pyramid(points, cs)
#            dofvals = numpy.concatenate([c.evaluatedofs(elt.values) for c in cs if c is not None], axis=0)
#            ndof, nfn = dofvals.shape
#            numpy.testing.assert_array_almost_equal(dofvals, numpy.hstack((numpy.eye(ndof), numpy.zeros((ndof,nfn - ndof )))))
                