'''
Created on Aug 30, 2010

@author: joel
'''
import unittest
import numpy as np
from pypyr.mesh import buildcubemesh
from pypyr.utils import pyramidquadrature, squarequadrature, trianglequadrature, uniformcubepoints
from pypyr.elements import H1Elements, HcurlElements, HdivElements, L2Elements  
from pypyr.assembly import SymmetricSystem, AsymmetricSystem

class TestSymmetric(unittest.TestCase):
    
    def testSymmetry(self):
        tag = "B1"
        for k in range(1,3):
            quadrule = pyramidquadrature(k+1)
            for N in range(1,3):
                for elements in [H1Elements(k), HcurlElements(k), HdivElements(k)]:
                    system = SymmetricSystem(elements, quadrule, lambda m: buildcubemesh(N, m, tag), [tag])
                    for deriv in [False, True]:
                        SM = system.systemMatrix(deriv)
                        g = lambda x: np.zeros((len(x),1))
                        S, SIBs, Gs = system.processBoundary(SM, {tag:g})  
                        np.testing.assert_array_almost_equal(SM.todense(), SM.transpose().todense())  
                        np.testing.assert_array_almost_equal(S.todense(), S.transpose().todense())
                        
class TestAsymmetric(unittest.TestCase):
    def testSymmetry(self):
        tag = "B1"        
        for k in range(1,3):
            quadrule = pyramidquadrature(k+1)            
            for N in range(1,3):
                hdivelt1 = HdivElements(k)
                l2elt1 = L2Elements(k)
                hdivelt2 = HdivElements(k)
                l2elt2 = L2Elements(k)
                system = AsymmetricSystem(hdivelt1, l2elt1, quadrule, lambda m:buildcubemesh(N,m,tag), [], [])
                systemt = AsymmetricSystem(l2elt2, hdivelt2, quadrule, lambda m:buildcubemesh(N,m,tag), [], [])
                SM = system.systemMatrix(True, False)
                StM = system.transpose().systemMatrix(False, True)
                SMt = systemt.systemMatrix(False, True)
                np.testing.assert_array_almost_equal(SM.todense(), SMt.transpose().todense())  
                np.testing.assert_array_almost_equal(StM.todense(), SMt.todense())  
                
                
            