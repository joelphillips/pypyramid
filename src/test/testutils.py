'''
Created on Aug 20, 2010

@author: joel
'''
import unittest

from pypyr.utils import *

class TestQuadrature(unittest.TestCase):
    
    def testPyQuad(self):
        n = 3
        x,w = pyramidquadrature(n)
        # integrate 1
        self.assertAlmostEqual(sum(w), 1.0/3)
        # integrate 1-z
        self.assertAlmostEqual(numpy.dot((1-x[:,2]), w), 1.0/4)
        # integrate z
        self.assertAlmostEqual(numpy.dot(x[:,2], w), 1.0/12)
        # integrate x
        self.assertAlmostEqual(numpy.dot(x[:,0], w), 1.0/8)
        # integrate y
        self.assertAlmostEqual(numpy.dot(x[:,1], w), 1.0/8)
     
    def testSquareQuad(self):
        n = 3
        x,w = squarequadrature(n)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertAlmostEqual(numpy.dot(x[:,0], w), 1.0/2)
        self.assertAlmostEqual(numpy.dot(x[:,1], w), 1.0/2)
        self.assertAlmostEqual(numpy.dot(x[:,0]*x[:,1], w), 1.0/4)
           