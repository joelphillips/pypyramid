'''
Created on Aug 20, 2010

@author: joel
'''
from pypyr.functions import *
from pypyr.mappings import Pullback

class DiffForm(object):
    def __init__(self, cpts, derivs, d2 = False):
        self.cpts = cpts
        self.ncpts = len(cpts)
        self.nfns = cpts[0].nfns
        self.derivs = derivs
        self.d2 = d2

    def values(self, p):
        return numpy.concatenate([cpt.values(p)[...,numpy.newaxis] for cpt in self.cpts], axis = 2)
    
    def D(self):
        if self.d2: dcpts = [ZeroFns(self.nfns)] * len(self.derivs[0]) 
        else: dcpts = [LinearComb([self.cpts[i].deriv(j) for (i,j) in zip(*numpy.nonzero(d))], d[numpy.nonzero(d)]) for d in self.derivs[0]]
        return DiffForm(dcpts, self.derivs[1:], True)
    
class CatDiffForm(object):
    def __init__(self, dfs):
        self.dfs = dfs
        self.ncpts = dfs[0].ncpts
        self.nfns = sum([df.nfns for df in dfs])
    
    def values(self, p):
        return numpy.concatenate([df.values(p) for df in self.dfs], axis=1)
    
    def D(self):
        return CatDiffForm([df.D() for df in self.dfs])
    
                  
class MapDiffForm():        
    def __init__(self, underlying, f, weights):
        self.underlying = underlying
        self.f = f
        self.weights = weights
        self.values = Pullback(f, weights)(underlying.values) # sexy python construct
        self.ncpts = underlying.ncpts
        self.nfns = underlying.nfns
    
    def D(self):
        return MapDiffForm(self.underlying.D(), self.f, self.weights[1:])
    
        
grad = numpy.array([[[1,0,0]],[[0,1,0]],[[0,0,1]] ])
c1 = numpy.array([[0,0,0],[0,0,1],[0,-1,0]]) # represents d_2 u_3 - d_3 u_2  
curl = numpy.array([c1[numpy.ix_(i,i)] for i in [[0,1,2],[2,0,1],[1,2,0]] ])
div = numpy.array([numpy.eye(3)])

derham = [grad, curl, div]    