'''
Created on Aug 20, 2010

@author: joel
'''
import unittest
import numpy
from pypyr.mappings import psijac, psiinvjac, psidet, applyweights, derham3dweights, buildaffine

class TestPullBack(unittest.TestCase):
        
    def testPsiJacobian(self):
        p = numpy.array([[0,0,0], [0,0.5,0.5], [0.5,0.2,0.5], [0.5,0.5,0]])
        js = psijac(p)
        jis = psiinvjac(p)
        for j, ji in zip(js, jis):
            numpy.testing.assert_array_almost_equal(numpy.dot(j, ji), numpy.eye(3))

    def testPullBack(self):
        p = numpy.array([[0,0,0], [0,0.5,0.5], [0.5,0.2,0.5], [0.5,0.5,0]])

        w = derham3dweights(psijac, psiinvjac, psidet)
        v0 = numpy.array([1,2,3,4]).reshape(-1,1)
        v0p = applyweights(w[0](p), v0)

        numpy.testing.assert_array_almost_equal(v0[...,numpy.newaxis],v0p)
        
        v1 = numpy.array([[0,1,0],[0,1,0],[1,0,0],[0,0,1]]) # normal vectors to the infinite pyramid at psi(p)
        v1p = applyweights(w[1](p), v1[:,numpy.newaxis,:])
        # check that they remain tangent after being mapped
        ns = numpy.array([[0,1,0], [0,1,1], [0.5,0,0.5], [0,0,1]])
        numpy.testing.assert_array_almost_equal(numpy.sum(numpy.abs(numpy.cross(v1p[:,0,:], ns)),axis=1), numpy.zeros(len(p)))
                
        # s=2 pullback preserves tangent vectors
        v2 = numpy.array([[[1,0,0],[0,0,1]], [[1,0,0],[0,0,1]], [[0,1,0],[0,0,1]], [[1,0,0],[0,1,0]]])
        v2p = applyweights(w[2](p), v2)
        
        numpy.testing.assert_array_almost_equal(numpy.sum(v2p * ns[:,numpy.newaxis,:], axis=2), numpy.zeros((len(p),2)))

class TestAffineMap(unittest.TestCase):
    
    N = 10 
    def test3to3(self):
        """ build an affine map to map 4 points in 3-space to 4 other points in 3-space """
        for _ in range(self.N):
            pf = numpy.random.random((4,3))
            pt = numpy.random.random((4,3))
            A = buildaffine(pf, pt)
            numpy.testing.assert_array_almost_equal(A.apply(pf), pt)
            numpy.testing.assert_array_almost_equal(A.applyinv(A.apply(pf)), pf)
        
    def test2to3(self):
        """ build an affine map to map 3 points in 2-space to 3 points in 3-space """
        for _ in range(self.N):
            pf = numpy.random.random((3,2))
            pt = numpy.random.random((3,3))
            A = buildaffine(pf, pt)
            numpy.testing.assert_array_almost_equal(A.apply(pf), pt)
    
    def test3to3missing(self):
        """ build an affine map to map 3 points in 3-space to 3 points in 3-space, preserving orthogonal vectors"""
        for _ in range(self.N):
            pf = numpy.random.random((3,3))
            pt = numpy.random.random((3,3))
            A = buildaffine(pf, pt)
            numpy.testing.assert_array_almost_equal(A.apply(pf), pt)
            # construct the orthogonal vector
            fognal = numpy.cross(pf[1] - pf[0], pf[2] - pf[0]) + pf[0]
            numpy.testing.assert_array_almost_equal(numpy.dot(A.apply(fognal) - pt[0], (pt[1:] - pt[0]).transpose()), [[0,0]])
            numpy.testing.assert_array_almost_equal(A.applyinv(A.apply(pf)), pf)
        
    
        