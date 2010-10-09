'''
Created on Aug 20, 2010

@author: joel
'''
import numpy
from numpy import newaxis

from pypyr.shapefunctions import buildRForms
from pypyr.mappings import buildaffine, mapbasedpullback, Pullback
from pypyr.timing import print_timing, Timer
   
class Element(object):
    """ A finite element.
    
    The shape functions are spanned by the pullbacks of diffform, and are dual to degrees.  degrees need not be a 
    complete dual basis, in which case the extra shape functions will be an orthogonal basis for the orthogonal complement
    of the space chosen to be dual to degrees.       
    """
    def __init__(self, diffform, degrees, pullback, invmap, indices):
        
        from numpy.linalg import qr, solve
        t = Timer().start()
        self.indices = numpy.array(indices)
        self.pullback = pullback
        self.invmap = invmap
        self.uvalues = pullback(diffform.values)
        if pullback.next() is not None: self.uderivs = pullback.next()(diffform.D().values)
        t.split("Initialised stuff")
        
        # We want to find M which is a right inverse for DV - the face degrees of freedom applied to our 
        # shape functions.  The values we return will be uvalues(p) * [M, N] where N extends M to full rank  
        
        # Note that if (v_i, v_j) = delta_ij for some inner product then if x_k y_k = 0 
        # then (v_ix_i, v_jy_j) = 0.  This means that if we extend M orthogonally, the shape-functions
        # associated with the volume degrees will be (.,.)-orthogonal to all the other shape functions.
        
        # So, lets write [DV^t,0] = [Q, Q_0].R  Then take M^t = R^{-1} Q^t and N = Q_0 
        
#        DV = numpy.vstack([d.evaluatedofs(self.uvalues) for d in degrees if d is not None])
        if len(degrees):
            DV = DegreeSet(degrees).evaluatedofs(self.uvalues) 
            
            t.split("Evaluate degrees")
            ndofs = DV.shape[0]
            nfns = DV.shape[1]
            
            Z = numpy.zeros((nfns - ndofs, nfns))
    #        print DV
            Q, R = qr(numpy.vstack((DV, Z)).transpose())
            t.split("QR decomposition")
            QQt = Q.transpose()[:ndofs,:]
            N =  Q[:,ndofs:]
            RR = R[:ndofs,:ndofs]
            Mt = solve(RR, QQt)
            t.split("Matrix solve")
            self.M = numpy.hstack((Mt.transpose(), N))
        else: self.M = numpy.eye(diffform.nfns)
#        t.show()
        
    def applyM(self, uv):
        # get the values, multiply by M and put the 2nd dimension (which indexes the shape functions) back in position
        n = len(uv.shape)
        uvM = numpy.tensordot(uv, self.M, ([1], [0])).transpose([0,n-1] + range(1,n-1))
        return uvM 

    def values(self, p):
        return self.applyM(self.uvalues(p))
        
    def derivs(self, p):        
        return self.applyM(self.uderivs(p))
        
    def maprefvalues(self, rv, rp, deriv=True):
        p = self.invmap(rp)
        pb = self.pullback.next() if deriv else self.pullback
        return self.applyM(pb.apply(p, rv))
    

class ExternalDegree(object):
    def __init__(self, pullback, points, dofs, indices):
        self.pullback = pullback
        self.points = points
        self.dofs = dofs
        self.indices = indices

    def evaluatedofs(self, f):        
        fvals = self.pullback(f)(self.points)        
        n = len(fvals.shape)
        return numpy.tensordot(self.dofs.reshape((-1, len(fvals))+fvals.shape[2:]), fvals, ([1]+range(2,n), [0]+range(2,n)))        

class DegreeSet(object):
    def __init__(self, degrees):
        self.points = numpy.vstack([d.pullback.map(d.points) for d in degrees if d is not None])
        self.ids = numpy.cumsum([0]+[len(d.dofs) for d in degrees if d is not None])
        self.dofs = [d.dofs for d in degrees if d is not None]
        self.indices = numpy.concatenate([d.indices for d in degrees if d is not None])
        self.weights = numpy.concatenate([d.pullback.weights[0](d.points) for d in degrees if d is not None], axis=0)
        
    
    def evaluatedofs(self, f):
        from mappings import applyweights
        fvals = applyweights(self.weights, f(self.points))
        n = len(fvals.shape)
        dofvals = []
        for dof, (id0, id1) in zip(self.dofs, zip(self.ids[:-1], self.ids[1:])):
            dofvals.append(numpy.tensordot(dof.reshape((-1, id1-id0)+fvals.shape[2:]), fvals[id0:id1], ([1]+range(2,n), [0]+range(2,n))))
        return numpy.vstack(dofvals)

refpyramid = numpy.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0,0,1]])
refpoints2d = numpy.vstack((numpy.zeros((1,2)), numpy.eye(2)))
refpoints1d = numpy.array([[0],[1]])

class ElementFactory(object):
    """ Base class for creating elements and degrees of freedom 
    
    To use: implement self.Pullback(map) and override vertex, edge, triangle, quad as necessary"""
     
    index = 0
    
    def createDegreeMethod(self, refvertices, refpoints, dofs):
        """ Returns a method that will create degrees of freedom"""
        if len(dofs) == 0: return lambda vertices: None
        def createExternalDegree(vertices):
            if len(refvertices) == 0: 
                pullback = Pullback(lambda p: vertices[numpy.ix_([0])])
            else: 
                map = buildaffine(refvertices, vertices[0:len(refvertices)])
                pullback = self.createpullback(map)
#                print "vertices", vertices
#                print "affine", map.linear
            indices = range(self.index, self.index + len(dofs))
            self.index += len(dofs)
            return ExternalDegree(pullback, refpoints, dofs, indices)
        return createExternalDegree
    
    def pyramid(self, points, degrees):
        indices = sum([d.indices for d in degrees if d is not None], [])
        ninternal = self.pyramidform.nfns - len(indices)
        indices.extend(range(self.index, self.index + ninternal))
        self.index += ninternal
        map = buildaffine(points[[0,1,3,4]], refpyramid[[0,1,3,4]])        
        pullback = self.createpullback(map)
        elt = Element(self.pyramidform, degrees, pullback, map.applyinv, indices)
        
        return elt

    
    def vertex(self,point): return None
    def edge(self,points): return None
    def triangle(self,points): return None
    def quad(self,points): return None
        
def edgepoints(k):
    return numpy.linspace(0,1,k+1,False)[1:].reshape(-1,1)

def squarepoints(k, extra=0):
    g = numpy.mgrid[0:k,0:k].reshape(2,-1)
    ep = edgepoints(k)
    return numpy.hstack((ep[g[0]], ep[g[1]])+(numpy.zeros((g.shape[1],1)),)*extra)

def trianglepoints(k,extra=0):
    g = numpy.mgrid[0:k+1,0:k+1].reshape(2,-1)
    gg = g[:,numpy.sum(g,axis=0) < k]
    ep = edgepoints(k+1)
    return numpy.hstack((ep[gg[0]], ep[gg[1]])+(numpy.zeros((gg.shape[1],1)),)*extra)

class H1Elements(ElementFactory):
    def __init__(self, k):
        self.pyramidform = buildRForms(k)[0]
        self.createpullback = lambda map : mapbasedpullback(map, 0) 
        self.vertex = self.createDegreeMethod([], numpy.zeros((1,0)), numpy.eye(1))

        if k >=2:
            self.edge = self.createDegreeMethod(refpoints1d, edgepoints(k-1), numpy.eye(k-1))
            self.quad = self.createDegreeMethod(refpoints2d, squarepoints(k-1), numpy.eye((k-1)*(k-1)))
        if k >=3:
            self.triangle = self.createDegreeMethod(refpoints2d, trianglepoints(k-2), numpy.eye((k-1)*(k-2)/2))

class HdivElements(ElementFactory):
    def __init__(self, k):
        from numpy import eye, concatenate, zeros
        self.createpullback = lambda map: mapbasedpullback(map, 2)
        self.pyramidform = buildRForms(k)[2]
        ntri = k * (k+1) / 2
        nquad = k*k
        self.triangle = self.createDegreeMethod(refpyramid[[0,3,1]], trianglepoints(k,1), concatenate((zeros((ntri, ntri, 2)), -eye(ntri)[...,newaxis]), axis=2))
        self.quad = self.createDegreeMethod(refpyramid[[0,3,1]], squarepoints(k,1), concatenate((zeros((nquad, nquad, 2)), -eye(nquad)[...,newaxis]), axis=2))
        
class L2Elements(ElementFactory):
    def __init__(self, k):
        self.createpullback = lambda map: mapbasedpullback(map, 3)
        self.pyramidform = buildRForms(k)[3]
                                                         
