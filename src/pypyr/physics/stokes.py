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

def stokes2(k, meshevents, v, points):
    vortelts1 = pe.HcurlElements(k)
    vortelts2 = pe.HcurlElements(k)
    velelts1 = pe.HdivElements(k)
    velelts2 = pe.HdivElements(k)
    pressureelts1 = pe.L2Elements(k)
    
    quadrule = pu.pyramidquadrature(k+1)
    
    Asys = pa.SymmetricSystem(vortelts1, quadrule, meshevents, [])
#    Bsys = pa.AsymmetricSystem(velelts1, vortelts2, quadrule, meshevents, [bdytag], [])
    BsysT = pa.AsymmetricSystem(vortelts2, velelts1, quadrule, meshevents, [], [openbdytag])
    Csys = pa.AsymmetricSystem(pressureelts1,velelts2, quadrule, meshevents, [], [openbdytag])
    
    A = Asys.systemMatrix(False)
    BT = BsysT.systemMatrix(True, False)
    C = Csys.systemMatrix(False, True)
    
    vv = lambda x: np.tile(v,(len(x), 1))[:,np.newaxis,:]
    vn = lambda x,n: np.tensordot(n,v,([1],[1]))
#    vt = lambda x,n: (v - vn(x,n)*n)[:,np.newaxis,:]
    vc = lambda x,n: np.cross(v, n)[:, np.newaxis, :]
    
    BTI, BTE, BTGs = BsysT.processBoundary(BT, {openbdytag:vv})
    CI, CE, CGs = Csys.processBoundary(C, {openbdytag:vv})
    
    P = Csys.loadVector(lambda x: np.ones((len(x),1,1)))
#    print "P ",P
    
    Gt = Asys.boundaryLoad({openbdytag: vc}, pu.squarequadrature(k+1), pu.trianglequadrature(k+1), False)

#    print "Gt ",Gt
    print A.shape, BT.shape, C.shape, BTI.shape, BTE[openbdytag].shape, BTGs[openbdytag].shape, CI.shape

    AL = Gt[openbdytag] + BTE[openbdytag] * BTGs[openbdytag]
 #   print "AL ",AL
    CL = -CE[openbdytag] * CGs[openbdytag]
    
    nvort = A.get_shape()[0]
    nvel = BTI.get_shape()[1]
    
#    S = ss.bmat([[A, -BTI, None],[-BTI.transpose(), None, CI.transpose()],[None, CI, None]])
#    L = np.vstack((AL,np.zeros((nvel,1)), CL))
    S = ss.bmat([[A, -BTI, None, None],[-BTI.transpose(), None, CI.transpose(), None],[None, CI, None, P], [None,None,P.transpose(), None]])
    L = np.vstack((AL,np.zeros((nvel,1)), CL, np.zeros((1,1))))
    X = ssl.spsolve(S, L)
    U = X[nvort:(nvort + nvel)]
#    print "X",X
#    print "U", U
#    print "BTGs", BTGs
#    
    u = BsysT.evaluate(points, U, BTGs, False)
#    uu = Asys.evaluate(points, np.eye(nvort)[-2], {}, False)
#    uu = BsysT.evaluate(points, U, {}, False)
    uu = BsysT.evaluate(points, np.zeros_like(U), BTGs, False)
#    print np.hstack((points, u))
#    print u
    return u, uu


def stokespressure(k, meshevents, pressures, closedtag, points):
    vortelts1 = pe.HcurlElements(k)
    vortelts2 = pe.HcurlElements(k)
    velelts1 = pe.HdivElements(k)
    velelts2 = pe.HdivElements(k)
    pressureelts1 = pe.L2Elements(k)
    
    quadrule = pu.pyramidquadrature(k+1)
    
    Asys = pa.SymmetricSystem(vortelts1, quadrule, meshevents, [])
#    Bsys = pa.AsymmetricSystem(velelts1, vortelts2, quadrule, meshevents, [bdytag], [])
    BsysT = pa.AsymmetricSystem(vortelts2, velelts1, quadrule, meshevents, [], [closedtag])
    Csys = pa.AsymmetricSystem(pressureelts1,velelts2, quadrule, meshevents, [pressures.keys()], [closedtag])
    
    A = Asys.systemMatrix(False)
    BT = BsysT.systemMatrix(True, False)
    C = Csys.systemMatrix(False, True)
    
    v0 = lambda x,n: np.zeros_like(x)
    vt = lambda x,n: np.zeros_like(x)
    
    BTI, BTE, BTGs = BsysT.processBoundary(BT, {closedtag:v0})
    CI, CE, CGs = Csys.processBoundary(C, {closedtag:v0})
    
    P = Csys.loadVector(lambda x: np.ones((len(x),1,1)))
#    print "P ",P
    alltags = pressures.keys() + [closedtag]
    
    Gt = Asys.boundaryLoad(dict([(tag,vt) for tag in alltags]), pu.squarequadrature(k+1), pu.trianglequadrature(k+1), False)

#    print "Gt ",Gt
    print A.shape, BT.shape, C.shape, BTI.shape, BTE[openbdytag].shape, BTGs[openbdytag].shape, CI.shape

    AL = Gt[openbdytag] + BTE[openbdytag] * BTGs[openbdytag]
 #   print "AL ",AL
    CL = -CE[openbdytag] * CGs[openbdytag]
    
    nvort = A.get_shape()[0]
    nvel = BTI.get_shape()[1]
    
#    S = ss.bmat([[A, -BTI, None],[-BTI.transpose(), None, CI.transpose()],[None, CI, None]])
#    L = np.vstack((AL,np.zeros((nvel,1)), CL))
    S = ss.bmat([[A, -BTI, None, None],[-BTI.transpose(), None, CI.transpose(), None],[None, CI, None, P], [None,None,P.transpose(), None]])
    L = np.vstack((AL,np.zeros((nvel,1)), CL, np.zeros((1,1))))
    X = ssl.spsolve(S, L)
    U = X[nvort:(nvort + nvel)]
#    print "X",X
#    print "U", U
#    print "BTGs", BTGs
#    
    u = BsysT.evaluate(points, U, BTGs, False)
#    uu = Asys.evaluate(points, np.eye(nvort)[-2], {}, False)
#    uu = BsysT.evaluate(points, U, {}, False)
    uu = BsysT.evaluate(points, np.zeros_like(U), BTGs, False)
#    print np.hstack((points, u))
#    print u
    return u, uu
    
def stokescubemesh(n, mesh):
    """ Produces the events to construct a mesh consisting of n x n x n cubes, each divided into 6 pyramids"""
    l = np.linspace(0,1,n+1)
    idxn1 = np.mgrid[0:n+1,0:n+1,0:n+1].reshape(3,-1).transpose()
    openbdy = []
    closedbdy = []
    for i in idxn1: 
        mesh.addPoint(tuple(i), l[i])
        if (i==0).any() or (i==n).any(): openbdy.append(tuple(i)) 
#        if i[0]==0 or i[1]==n: openbdy.append(tuple(i)) 
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

class MeshPlotter(pm.MeshBase):
    triangles = []
    def addPyramid(self, pointids):
        pointids = np.array(pointids, dtype=object)
        self.triangles.extend([pointids[t] for t in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]])
    
    def plot(self, fig):
        x,y,z = zip(*self.getPoints(np.array(self.triangles).flatten()))
        emm.triangular_mesh(x,y,z, np.arange(len(self.triangles)*3).reshape(-1,3), representation = 'wireframe', figure = fig, color=(0,0,0), line_width=0.5)
        
        
if __name__ == "__main__":
    k = 2
    N = 6
    points = pu.uniformcubepoints(8)
    v = [[1,1,1]]
#    v = [[0,0,0]]
    meshevents = lambda m: stokescubemesh(N, m)
    mp = MeshPlotter()
    meshevents(mp)
    u, uu = stokes2(k,meshevents,np.array(v), points)
    pt = points.transpose()
    ut = u.transpose()
#    print ut
    emm.figure(bgcolor=(1,1,1))

    emm.quiver3d(pt[0],pt[1],pt[2], ut[0],ut[1],ut[2])

#    emm.flow(pt[0],pt[1],pt[2], ut[0],ut[1],ut[2])
    mp.plot(emm.gcf())
#    emm.figure()
#    uut = uu.transpose()
#    emm.quiver3d(pt[0],pt[1],pt[2], uut[0],uut[1],uut[2])
    emm.show()



#    
#    
#def stokes(k, meshevents, v, points):
#    vortelts = pe.HcurlElements(k)
#    potelts = pe.HcurlElements(k)
#    potelts2 = pe.HcurlElements(k)
#    lagelts = pe.H1Elements(k)
#    
#    quadrule = pu.pyramidquadrature(k+1)
#    
#    Asys = pa.SymmetricSystem(vortelts, quadrule, meshevents, [])
#    Bsys = pa.SymmetricSystem(potelts, quadrule, meshevents, [])
#    Csys = pa.AsymmetricSystem(lagelts, potelts2, quadrule, meshevents, [])
#    
#    A = Asys.systemMatrix(False)
#    B = Bsys.systemMatrix(True)
#    C = Csys.systemMatrix(True, False)
#    
#    vn = lambda x,n: np.tensordot(n,v,([1],[1]))
#    vt = lambda x,n: (v - vn(x,n)*n)[:,np.newaxis,:]
#    
#    G = Asys.boundaryLoad({bdytag: vt}, pu.squarequadrature(k+1), pu.trianglequadrature(k+1), False)
#    b = Csys.boundaryLoad({bdytag: vn}, pu.squarequadrature(k+1), pu.trianglequadrature(k+1), False)
#
#    gg = G[bdytag]
#    bb = b[bdytag]
#    
#    ng = len(gg)
#    nb = len(bb)
#    print A.shape, B.shape, C.shape, gg.shape, bb.shape, ng, nb
#    
#    S = ss.bmat([[A, -B, None, None],[-B, None, -C.transpose(), None],[None, -C, None, np.ones((nb,1))],[None,None,np.ones((1,nb)), None]])
#    L = np.vstack((gg,np.zeros_like(gg), -bb, np.zeros((1,1))))
#    X = ssl.spsolve(S, L)
#    print "gg", gg
#    print "bb", bb
#    print "X",X
#    
#    u = Bsys.evaluate(points, X[ng:2*ng], {}, True)
#
#    return u
#    
#                