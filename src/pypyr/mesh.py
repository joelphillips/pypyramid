'''
A collection of classes representing things that happen at the level of the mesh.  A simple event-based model is used
to define the mesh; see buildcubemesh for an example of how to produce the events.

Created on Aug 23, 2010

@author: joel
'''

import numpy
import numpy as np
from numpy import newaxis
from pypyr.elements import DegreeSet
from pypyr.timing import print_timing

class MeshBase(object):
    """ Utility base class for classes that consume mesh events.  Knows about points and boundaries"""
    def __init__(self):
        self.__points = {}
        self.boundaries = {}
    
    def addPoint(self, id, point):
        self.__points[id] = point
    
    def getPoints(self, ids):
        return numpy.array([self.__points[id] for id in ids])

    def addBoundary(self, id, pointids):
        self.boundaries[id] = pointids
        
class Basis(MeshBase):
    """ A basis that lives on a mesh."""
    
    def __init__(self, elementfactory):
        """ elementfactory: used to construct an element for each subdomain of the mesh"""
        super(Basis,self).__init__()
        self.elements = []
        self.elementfactory = elementfactory
        self.degrees = {}
        
        
    def getBoundary(self, id):
        idset = set(map(tuple, self.boundaries[id]))
        bdyds = []
        for ids, d in self.degrees.iteritems():
            if idset.issuperset(ids): bdyds.append(d)
        return DegreeSet(bdyds) if len(bdyds) else None
    
    
    def getElementValues(self, refpoints, deriv=True):
        """ Generator method - to keep memory happy"""
        refform = self.elementfactory.pyramidform
        if deriv: refform = refform.D()
        refvals = refform.values(refpoints)
        for e in self.elements:
            yield e.maprefvalues(refvals, refpoints, deriv)
    
    def getIndices(self):
        return numpy.concatenate([e.indices for e in self.elements])
        
    def addPyramid(self, pointids):
        """ Add a pyramid.
        
        vertexids should be a list of vertices - base first, then the summit. """
        
        pds = []
        pointids = numpy.array(pointids, dtype=object)
        vertexpoints = [[p] for p in pointids]
        edgepoints = [pointids[e] for e in [[0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4]]]
        trianglepoints = [pointids[t] for t in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]]
        quadpoints = [pointids[0:4]]
        pds.extend(self._degrees(vertexpoints, self.elementfactory.vertex))
        pds.extend(self._degrees(edgepoints, self.elementfactory.edge))
        pds.extend(self._degrees(trianglepoints, self.elementfactory.triangle))
        pds.extend(self._degrees(quadpoints, self.elementfactory.quad))  
        self.elements.append(self.elementfactory.pyramid(self.getPoints(pointids), pds))     
    
    def _degrees(self, pointlists, factory):
        ds = []
        for id in pointlists:
            tid = tuple(sorted(id))
            try:
                d = self.degrees[tid]
            except KeyError:
                d = factory(self.getPoints(tid))
                self.degrees[tid] = d
            if d is not None: ds.append(d)
        return ds

class ElementQuadrature(MeshBase):
    """ What are the quadrature rules associated with each element?
    
    Obviously, there's a bit of duplication with the Basis class - we're recalculating the affine maps here, but hopefully that's not
    too expensive."""
    def __init__(self):
        super(ElementQuadrature, self).__init__()
        self.maps = []
        
    def addPyramid(self, pointids): 
        from elements import refpyramid
        from mappings import buildaffine
        points = self.getPoints(pointids) 
        self.maps.append(buildaffine(refpyramid[[0,1,3,4]], points[[0,1,3,4]]))
    
    def getWeights(self, refpoints, refweights):
        for m in self.maps:
            yield numpy.abs(m.dets(refpoints)) * refweights
    
    def getQuadPoints(self, refpoints):
        for m in self.maps:
            yield m.apply(refpoints)
    
    def numElements(self):
        return len(self.maps)
    
class BoundaryQuadrature(MeshBase):
    """ What are the quadratures associated with the external boundary of each element?"""
    
    def __init__(self):
        super(BoundaryQuadrature, self).__init__()
        self.pyramidpoints = []
    
    def addPyramid(self, pointsids):
        self.pyramidpoints.append(numpy.array(pointsids, dtype=object))
    
    def getQuadratures(self, tag, squarequad, trianglequad):
        """ For a boundary identified by tag, return a list of quadrature points, weights and normals for each element """
        from mappings import buildaffine
        from elements import refpyramid
        bdy = set(self.boundaries[tag].__iter__())
        
        sx,sw = squarequad
        sx = np.hstack((sx, np.zeros((len(sx),1))))
        
        tx,tw = trianglequad
        tx = np.hstack((np.zeros((len(tx),1))))
        
        for ppoints in self.pyramidpoints:
            x = []
            w = []
            normals = []
            pyramidcentre = np.sum(self.getPoints(ppoints), axis=0) / 5.0
            for tripoints in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]:
                if bdy.issuperset(ppoints[tripoints]):
                    m = buildaffine(refpyramid[[0,1,4]], self.getPoints(ppoints[tripoints]))
                    x.append(m.apply(tx))
                    w.append(numpy.abs(m.dets(tx))*tw)
                    x0 = m.apply(refpyramid[0])[0]
                    n = m.apply(refpyramid[3])[0] - x0 
                    normals.append(np.tile(-n * np.sign(np.dot(pyramidcentre - x0, n)), (len(tx),1) ))
            if bdy.issuperset(ppoints[0:4]):
                m = buildaffine(refpyramid[[0,1,3]], self.getPoints(ppoints[[0,1,3]]))
                x.append(m.apply(sx))
                w.append(numpy.abs(m.dets(sx)) * sw)
                x0 = m.apply(refpyramid[0])[0]
                n = m.apply(refpyramid[4])[0] - x0
#                print pyramidcentre, x0, n, -n * np.sign(np.dot(pyramidcentre - x0, n))
                normals.append(np.tile(-n * np.sign(np.dot(pyramidcentre - x0, n)), (len(sx),1) ))
                
            if x: yield numpy.vstack(x), numpy.concatenate(w), numpy.vstack(normals)
            else: yield numpy.zeros((0,3)), numpy.zeros((0,0)), numpy.zeros((0,3))
    
class ElementFinder(MeshBase):
    """ Used to determine which element a point lives in.  
    
    Uses an order 1 HdivElement to get normals to each element, which is cute, but duplicates BoundaryQuadrature a bit"""    
    def __init__(self):
        from elements import HdivElements
        MeshBase.__init__(self)
        self.hdivdegs = HdivElements(1)
        self.centres = []
        self.degrees = []
            
    def addPyramid(self, pointids):
        pds = []
        points = self.getPoints(pointids) 
#        print "addPyramid",points
        pds.extend([self.hdivdegs.triangle(points[t]) for t in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]])
        pds.append(self.hdivdegs.quad(points[0:4]))
        self.degrees.append(DegreeSet(pds))
        self.centres.append(numpy.sum(points, axis=0)/5.0)        
        
    def elementPointMap(self, points):
        """ given a collection of points, return an element->point map"""

        etop = []#[[] for _ in range(len(self.degrees))]
        unassigned = numpy.ones(len(points), dtype=bool)
        for centre, degrees in zip(self.centres, self.degrees):
            fpoints = lambda p: points[newaxis,:,:] - p[:,newaxis,:]
            fcentre = lambda p: centre.reshape(1,1,3) - p[:,newaxis,:]
                        
            cdofs = numpy.vstack(degrees.evaluatedofs(fcentre))
            pdofs = numpy.vstack(degrees.evaluatedofs(fpoints))
            ps = numpy.where(numpy.logical_and(numpy.sum(pdofs * numpy.sign(cdofs) >= 0,axis=0)==5, unassigned))
            etop.append(ps[0])
            unassigned[ps[0]] = False
        return etop
            
def buildcubemesh(n, mesh, boundarytag = None):
    """ Produces the events to construct a mesh consisting of n x n x n cubes, each divided into 6 pyramids"""
    from numpy import mgrid
    l = numpy.linspace(0,1,n+1)
    idxn1 = numpy.mgrid[0:n+1,0:n+1,0:n+1].reshape(3,-1).transpose()
    bdy = []
    for i in idxn1: 
        mesh.addPoint(tuple(i), l[i])
        if (i==0).any() or (i==n).any(): bdy.append(tuple(i)) 
    if boundarytag: mesh.addBoundary(boundarytag, bdy)
    
    l12 = (l[1:] + 1.0*l[:-1])/2.0
    idxn = numpy.mgrid[0:n, 0:n, 0:n].reshape(3,-1).transpose()
    cornerids = mgrid[0:2,0:2,0:2].reshape(3,8).transpose()
    
    for i in idxn:
        id = tuple(i) + (1,)
        mesh.addPoint(id, l12[i])
        for basecorners in [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]:
            mesh.addPyramid(map(tuple, cornerids[basecorners] + i)+[id])
            
    return mesh
        