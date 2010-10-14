'''
Created on Oct 10, 2010

@author: joel
'''

import pypyr.elements as pe
import pypyr.utils as pu
import pypyr.assembly as pa
import pypyr.mesh as pm
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import enthought.mayavi.mlab as emm

openbdytag = 'OPEN'
closedbdytag = 'CLOSED'
bdytag = 'BDY'

def stokes(k, meshevents, v, points):
    vortelts = pe.HcurlElements(k)
    potelts = pe.HcurlElements(k)
    potelts2 = pe.HcurlElements(k)
    lagelts = pe.H1Elements(k)
    
    quadrule = pu.pyramidquadrature(k+1)
    
    Asys = pa.SymmetricSystem(vortelts, quadrule, meshevents, [])
    Bsys = pa.SymmetricSystem(potelts, quadrule, meshevents, [])
    Csys = pa.AsymmetricSystem(lagelts, potelts2, quadrule, meshevents, [])
    
    A = Asys.systemMatrix(False)
    B = Bsys.systemMatrix(True)
    C = Csys.systemMatrix(True, False)
    
    vx = lambda x,n: np.tile(v, (len(x),1,1))
    vn = lambda x,n: np.dot(n,v)
    
    G = Asys.boundaryLoad({bdytag: vx}, pu.squarequadrature(k+1), pu.trianglequadrature(k+1), False)
    b = Csys.boundaryLoad({bdytag: vn}, pu.squarequadrature(k+1), pu.trianglequadrature(k+1), False)

    gg = G[bdytag]
    bb = b[bdytag]
    
    ng = len(gg)
    nb = len(bb)
    print A.shape, B.shape, C.shape, gg.shape, bb.shape, ng, nb
    
    S = ss.bmat([[A, B, None, None],[B, None, C.transpose(), None],[None, C, None, np.ones((nb,1))],[None,None,np.ones((1,nb)), None]])
    L = np.vstack((gg,np.zeros_like(gg), bb, np.zeros((1,1))))
    X = ssl.spsolve(S, L)
    print X
    
    u = Bsys.evaluate(points, X[ng:2*ng], {}, True)
    print u
    return u
    
            
def stokescubemesh(n, mesh):
    """ Produces the events to construct a mesh consisting of n x n x n cubes, each divided into 6 pyramids"""
    l = np.linspace(0,1,n+1)
    idxn1 = np.mgrid[0:n+1,0:n+1,0:n+1].reshape(3,-1).transpose()
    openbdy = []
    closedbdy = []
    for i in idxn1: 
        mesh.addPoint(tuple(i), l[i])
        if (i==0)[1:].any() or (i==n)[1:].any(): closedbdy.append(tuple(i)) 
        if i[0]==0 or i[1]==n: openbdy.append(tuple(i)) 
    mesh.addBoundary(openbdytag, openbdy)
    mesh.addBoundary(closedbdytag, closedbdy)
    
    l12 = (l[1:] + 1.0*l[:-1])/2.0
    idxn = np.mgrid[0:n, 0:n, 0:n].reshape(3,-1).transpose()
    cornerids = np.mgrid[0:2,0:2,0:2].reshape(3,8).transpose()
    
    for i in idxn:
        id = tuple(i) + (1,)
        mesh.addPoint(id, l12[i])
        for basecorners in [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]:
            mesh.addPyramid(map(tuple, cornerids[basecorners] + i)+[id])
            
    return mesh

        
if __name__ == "__main__":
    k = 3
    N = 1
    points = pu.uniformcubepoints(8)
    u = stokes(k,lambda m: pm.buildcubemesh(N, m, bdytag),np.array([1,0,0]), points)
    pt = points.transpose()
    ut = u.transpose()
    emm.quiver3d(pt[0],pt[1],pt[2], ut[0],ut[1],ut[2])
    emm.show()
    