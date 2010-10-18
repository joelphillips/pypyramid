'''
Created on Aug 17, 2010

@author: joel
'''
import numpy
from pypyr.mesh import Basis, ElementFinder, ElementQuadrature, BoundaryQuadrature
import itertools as it
from pypyr.timing import *

def processIndices(basis, boundarytags):
    """ Given a basis (a collection of elements) and a set of boundaries, extract the internal and external degrees of freedom 
    
    returns: 
        I: a sparse matrix that maps each the local degrees of freedom for each element to their global indices
        boundaries: a map of tag->DegreeSet, which can be used to evaluate all the degrees on each boundary
        internalidx: ids of the internal degrees of freedom    
    """
    import scipy.sparse as ss
    indices = basis.getIndices()
    n = basis.elementfactory.index # = max(indices)+1
    I = ss.csr_matrix((numpy.ones_like(indices), indices, range(0,len(indices)+1)))
            
    idxflag = numpy.ones(n, dtype=bool)
    boundaries = {}
    for tag in boundarytags:
        bdy = basis.getBoundary(tag)
        boundaries[tag] = bdy
        if bdy: idxflag[bdy.indices] = False
    internalidx = numpy.nonzero(idxflag)[0]
    return I, boundaries, internalidx

def blockInnerProducts(quadweights, leftvalsiter, rightvalsiter, leftI, rightI):   
    """ Evaluate the inner product matrix
    
    returns a sparse matrix equal to leftI.transpose * L.transpose * quadweights * R * rightI
    where L and R are block diagonal matrices whose blocks are given by the iterables, leftvalsiter and rightvalsiter
    
    If the left or right vals have more than 2 dimensions, the extra dimensions are multiplied and summed (tensor-contracted),
    with broadcasting as necessary, i,e, this is an inner-product - it can't be used for a more general multiplication'    
    """
    import scipy.sparse as ss    
    data = []
    idx = []
    ip = [0]        
    for e, (leftvals, rightvals, weights) in enumerate(it.izip(leftvalsiter, rightvalsiter, quadweights)):
        if len(weights):
            lvs = len(leftvals.shape)
            rvs = len(rightvals.shape)
            vs = max(lvs,rvs)
            leftvals = leftvals.reshape(leftvals.shape + (1,)*(vs - lvs))
            rightvals = rightvals.reshape(rightvals.shape + (1,)*(vs - rvs))  
            lvw = leftvals * weights.reshape((-1,) + (1,)*(vs-1))
#            print lvw.shape, rightvals.shape
            data.append(numpy.tensordot(lvw, rightvals,  ([0]+range(2,vs), [0]+range(2,vs))))
            idx.append(e)
        ip.append(len(idx))
#    print e, idx, ip
    V = ss.bsr_matrix((data, idx, ip),dtype=float, shape=(leftI.shape[0],rightI.shape[0]))
    return leftI.transpose() * V * rightI  


class System(object):
    """ A System contains everything that's need to construct stiffness matrices and load vectors.  
        This is an abstract-ish class see SymmetricSystem and AsymmetricSystem for concrete implementations.
        
        Parameters:
            quadrule: a tuple of quadrature points and weights on the reference pyramid
            meshevents: A function that produces mesh events
            leftbasis, rightbasis: see pypyr.mesh.Basis
            leftindexinfo, rightindexinfo: see processIndices
    """
    def __init__(self, quadrule, meshevents, leftbasis, rightbasis, leftindexinfo, rightindexinfo):
        self.elementfinder = meshevents(ElementFinder())
        self.elementinfo = meshevents(ElementQuadrature())
        self.boundaryquad = meshevents(BoundaryQuadrature())
        self.refquadpoints, refweights = quadrule
        self.quadweights = list(self.elementinfo.getWeights(self.refquadpoints, refweights))
        self.leftbasis = leftbasis
        self.rightbasis = rightbasis
        self.leftI, self.leftbdys, self.leftintidx = leftindexinfo
        self.rightI, self.rightbdys, self.rightintidx = rightindexinfo

    def processSystem(self, leftvalsiter, rightvalsiter):
        """ Construct the (non-boundary aware) stiffness matrix """
        return blockInnerProducts(self.quadweights, leftvalsiter, rightvalsiter, self.leftI, self.rightI)
        
    def processBoundary(self, sysmat, tagtog):
        """ Split the stiffness matrix into the internal and external parts.  Evaluate boundary data
        
        sysmat: system matrix (which will come from processSystem()).  
        tagtog: dictionary of functions to evaluate on the boundar(y|ies)
        
        returns:
            internalSystem: S[I,I] where I is the internal degrees
            tagtoBoundarySystem: tag->S[I,E[tag]] where E[tag] gives the indices of the external degrees
            tagtogvals: g[tag] evaluated at the degrees of freedom associated with boundary "tag".
        
        Somewhat inefficient if there's a significant proportion of dofs on the boundary """
        
        SI = sysmat[self.leftintidx, :]
        internalSystem = SI[:,self.rightintidx]
        tagtogvals = {}
        tagtoBoundarySystem = {}
        for tag, bdy in self.rightbdys.iteritems():
            tagtogvals[tag] = bdy.evaluatedofs(tagtog[tag])
            tagtoBoundarySystem[tag] = SI[:,bdy.indices]
        
        return internalSystem, tagtoBoundarySystem, tagtogvals        
    
    def loadVector(self, f, deriv=False):
        """ Calculate the load vector for the internal shape functions """ 
        testvalsiter = self.leftbasis.getElementValues(self.refquadpoints, deriv)
        fvalsiter = it.imap(f, self.elementinfo.getQuadPoints(self.refquadpoints))
        return blockInnerProducts(self.quadweights, testvalsiter, fvalsiter, self.leftI, numpy.ones((self.elementinfo.numElements(), 1)))[self.leftintidx,:]
    
    def boundaryLoad(self, tagtog, squarequad, trianglequad, deriv=False):
        """ Calculate the load vector based on a boundary integral, e.g. for Dirichlet data in the dual formulation of the mixed laplacian"""
        tagtogsys = {}
        for tag, g in tagtog.iteritems():
            x,w,n = zip(*self.boundaryquad.getQuadratures(tag, squarequad, trianglequad)) 
#            print map(g,x,n)          
#            print map(lambda e,p: 0 if len(p) is 0 else e.values(p), self.leftbasis.elements, x)
            fvalsiter = it.imap(g, x, n)
            testvalsiter = it.imap(lambda e,p: 0 if len(p) is 0 else e.values(p), self.leftbasis.elements, x)
            tagtogsys[tag] = blockInnerProducts(w, testvalsiter, fvalsiter, self.leftI, numpy.ones((self.elementinfo.numElements(), 1)))
        return tagtogsys                

    def evaluate(self, points, U, tagtoG = {}, deriv=False):
        """ Evaluate a solution given by the coefficients of the internal degrees, U, at specified points.  
        
        tagtoG should be the coefficients for the external degrees""" 
        
        UG = numpy.zeros(self.rightbasis.elementfactory.index)
        UG[self.rightintidx] = U
        for tag, G in tagtoG.iteritems():
            UG[self.rightbdys[tag].indices] = G
        etop = self.elementfinder.elementPointMap(points)
        UGvals = numpy.zeros((len(points), self.rightbasis.elements[0].ncpts))
        for e, pids in zip(self.rightbasis.elements, etop):
            if len(pids): 
                evals = e.derivs(points[pids]) if deriv else e.values(points[pids])
                UGvals[pids] += numpy.tensordot(evals, UG[e.indices], ([1],[0]))
        return UGvals    
         
class SymmetricSystem(System):
    """ A symmetric system"""
    def __init__(self, elements, quadrule, meshevents, boundarytags):
        self.basis = Basis(elements)
        meshevents(self.basis)
        indexinfo = processIndices(self.basis, boundarytags)
        System.__init__(self, quadrule, meshevents, self.basis, self.basis, indexinfo, indexinfo)
        self.elements = elements

    def systemMatrix(self, deriv):
        return super(SymmetricSystem, self).processSystem(*it.tee(self.basis.getElementValues(self.refquadpoints,deriv), 2))
                        
class AsymmetricSystem(System):
    """ An  Asymmetric system"""
    def __init__(self, leftelements, rightelements, quadrule, meshevents, leftboundarytags, rightboundarytags):
        leftbasis = Basis(leftelements)
        rightbasis = Basis(rightelements)
        meshevents(leftbasis)
        meshevents(rightbasis)
        super(AsymmetricSystem, self).__init__(quadrule, meshevents, leftbasis, rightbasis, processIndices(leftbasis, leftboundarytags), processIndices(rightbasis, rightboundarytags))

    def systemMatrix(self, leftderiv, rightderiv):
        leftvals = self.leftbasis.getElementValues(self.refquadpoints, leftderiv)
        rightvals = self.rightbasis.getElementValues(self.refquadpoints, rightderiv)
        return super(AsymmetricSystem, self).processSystem(leftvals, rightvals)
    
