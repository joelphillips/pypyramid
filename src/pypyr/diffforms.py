'''
Classes to represent spaces of differential forms

Each differential form class should have the following properties:
    ncpts: Number of components (i.e. (N, k) for a k-form on an N-manifold)
    nfns: Dimension of the space

Each differential forms class should have the following methods:
    values:  Evaluates all the differential forms at the provided points.  If df is a differential forms object, 
            df.values(p).shape == (len(p), df.nfns, df.ncpts)
    D: Returns the a differential forms object that represents the exterior derivatives of this object.
    
Exterior derivative information is provided by the derivs list.  The first entry in derivs should be a 3-tensor that describes 
the exterior derivative.  See DiffForm.D(self) and derham (at the end) to see how this works.     
    
Created on Aug 20, 2010

@author: joel
'''
from pypyr.functions import *
from pypyr.mappings import Pullback

class DiffForm(object):
    """ Start here.  Constructs a differential forms by specifying the components as function classes"""
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
    """ Concatenate 2 or more differential forms objects (think Helmholtz decomposition)
    def __init__(self, dfs):
        self.dfs = dfs
        self.ncpts = dfs[0].ncpts
        self.nfns = sum([df.nfns for df in dfs])
    
    def values(self, p):
        return numpy.concatenate([df.values(p) for df in self.dfs], axis=1)
    
    def D(self):
        return CatDiffForm([df.D() for df in self.dfs])
    
                  
class MapDiffForm():
    """ Represents the pullbacks of the underlying differential forms"""        
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