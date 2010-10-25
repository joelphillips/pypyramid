'''
Created on Oct 10, 2010

@author: joel
'''

import pypyr.elements as pe
import pypyr.utils as pu
import pypyr.assembly as pa
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

inputbdytag = 'INPUT'
outputbdytag = 'OUTPUT'
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
    BsysT = pa.AsymmetricSystem(vortelts2, velelts1, quadrule, meshevents, [], [bdytag])
    Csys = pa.AsymmetricSystem(pressureelts1,velelts2, quadrule, meshevents, [], [bdytag])
    
    A = Asys.systemMatrix(False)
    BT = BsysT.systemMatrix(True, False)
    C = Csys.systemMatrix(False, True)
    
    vv = lambda x: np.tile(v,(len(x), 1))[:,np.newaxis,:]
    vn = lambda x,n: np.tensordot(n,v,([1],[1]))
#    vt = lambda x,n: (v - vn(x,n)*n)[:,np.newaxis,:]
    vc = lambda x,n: np.cross(v, n)[:, np.newaxis, :]
    
    BTI, BTE, BTGs = BsysT.processBoundary(BT, {bdytag:vv})
    CI, CE, CGs = Csys.processBoundary(C, {bdytag:vv})
    
    P = Csys.loadVector(lambda x: np.ones((len(x),1,1)))
#    print "P ",P
    
    Gt = Asys.boundaryLoad({bdytag: vc}, pu.squarequadrature(k+1), pu.trianglequadrature(k+1), False)

#    print "Gt ",Gt
    print A.shape, BT.shape, C.shape, BTI.shape, BTE[bdytag].shape, BTGs[bdytag].shape, CI.shape

    AL = Gt[bdytag] + BTE[bdytag] * BTGs[bdytag]
 #   print "AL ",AL
    CL = -CE[bdytag] * CGs[bdytag]
    
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


def stokespressure(k, meshevents, pressures, points, countdofs = False, avpressure = False):
    vortelts1 = pe.HcurlElements(k)
    vortelts2 = pe.HcurlElements(k)
    velelts1 = pe.HdivElements(k)
    velelts2 = pe.HdivElements(k)
    pressureelts1 = pe.L2Elements(k)
    
    quadrule = pu.pyramidquadrature(k+1)
    
    Asys = pa.SymmetricSystem(vortelts1, quadrule, meshevents, [])
#    Bsys = pa.AsymmetricSystem(velelts1, vortelts2, quadrule, meshevents, [bdytag], [])
    BsysT = pa.AsymmetricSystem(vortelts2, velelts1, quadrule, meshevents, [], [closedbdytag])
    Csys = pa.AsymmetricSystem(pressureelts1,velelts2, quadrule, meshevents, [], [closedbdytag])
    
    A = Asys.systemMatrix(False)
    BT = BsysT.systemMatrix(True, False)
    C = Csys.systemMatrix(False, True)
    
    v0 = lambda x: np.zeros_like(x)[:,np.newaxis,:]
    vt = lambda x,n: np.zeros_like(x)[:,np.newaxis,:]
    
    BTI, BTE, BTGs = BsysT.processBoundary(BT, {closedbdytag:v0})
    CI, CE, CGs = Csys.processBoundary(C, {closedbdytag:v0})
    
    P = Csys.loadVector(lambda x: np.ones((len(x),1,1)))
#    print "P ",P
    alltags = pressures.keys() + [closedbdytag]
    
    Gt = Asys.boundaryLoad(dict([(tag,vt) for tag in alltags]), pu.squarequadrature(k+1), pu.trianglequadrature(k+1), False)
    Pv = Csys.transpose().boundaryLoad(pressures, pu.squarequadrature(k+1), pu.trianglequadrature(k+1), False)

#    print "Gt ",Gt
    print A.shape, BT.shape, C.shape, BTI.shape, map(np.shape, BTE.values()), map(np.shape, BTGs.values()), map(np.shape, Gt.values()), map(np.shape, Pv.values()), CI.shape

    AL = sum(Gt.values()) + BTE[closedbdytag] * BTGs[closedbdytag]
    BL = sum(Pv.values())
 #   print "AL ",AL
    CL = -CE[closedbdytag] * CGs[closedbdytag]
    
    nvort = A.get_shape()[0]
    nvel = BTI.get_shape()[1]
    print nvel
    
    if avpressure:
        S = ss.bmat([[A, -BTI, None, None],[-BTI.transpose(), None, CI.transpose(), None],[None, CI, None, P], [None,None,P.transpose(), None]])
        L = np.vstack((AL,BL, CL, np.zeros((1,1))))
    else:
        S = ss.bmat([[A, -BTI, None],[-BTI.transpose(), None, CI.transpose()],[None, CI, None]])
        L = np.vstack((AL,BL, CL))
#    L = np.vstack((AL,np.zeros((nvel,1)), CL))
#    print A.todense(), BTI.todense(), CI.todense()
#    print AL, BL, CL
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
    if countdofs:
        return u, len(X)
    return u, uu
    
def stokescubemesh(n, mesh):
    """ Produces the events to construct a mesh consisting of n x n x n cubes, each divided into 6 pyramids"""
    l = np.linspace(0,1,n+1)
    idxn1 = np.mgrid[0:n+1,0:n+1,0:n+1].reshape(3,-1).transpose()
    closedbdy = []
    inputbdy = []
    outputbdy = []
    for i in idxn1: 
        mesh.addPoint(tuple(i), l[i])
        if (i==0)[[1,2]].any() or (i==n)[[1,2]].any(): closedbdy.append(tuple(i)) 
        if i[0]==0: inputbdy.append(tuple(i))
        if i[0]==n: outputbdy.append(tuple(i)) 
    mesh.addBoundary(bdytag, closedbdy + inputbdy + outputbdy)
    mesh.addBoundary(closedbdytag, closedbdy)
    mesh.addBoundary(inputbdytag, inputbdy)
    mesh.addBoundary(outputbdytag, outputbdy)
    
    l12 = (l[1:] + 1.0*l[:-1])/2.0
    idxn = np.mgrid[0:n, 0:n, 0:n].reshape(3,-1).transpose()
    cornerids = np.mgrid[0:2,0:2,0:2].reshape(3,8).transpose()
    
    for i in idxn:
        id = tuple(i) + (1,)
        mesh.addPoint(id, l12[i])
        for basecorners in [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]:
            mesh.addPyramid(map(tuple, cornerids[basecorners] + i)+[id])
            
    return mesh

def pfn(p):
    return lambda x,n: (n * p)[:,np.newaxis,:]




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