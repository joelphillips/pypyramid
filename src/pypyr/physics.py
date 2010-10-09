'''
Created on Aug 28, 2010

@author: joel
'''
import numpy
import math
from pypyr.elements import H1Elements, HdivElements, L2Elements  
from pypyr.mesh import buildcubemesh
from pypyr.utils import pyramidquadrature, squarequadrature, trianglequadrature, uniformcubepoints
from scipy.sparse.linalg.dsolve.linsolve import spsolve 
from pypyr.timing import *
from pypyr.assembly import SymmetricSystem, AsymmetricSystem
import scipy.sparse.linalg as ssl 

def laplaceeigs(k,N,n):
    import scipy.linalg as sl
    tag = "B1"
    elements = H1Elements(k)
    quadrule = pyramidquadrature(k+1)
    system = SymmetricSystem(elements, quadrule, lambda m: buildcubemesh(N, m, tag), [tag])

    SM = system.systemMatrix(True)
    S, SIBs, Gs = system.processBoundary(SM, {tag:lambda p: numpy.zeros((len(p),1))})
    print S.shape
    MM = system.systemMatrix(False)
    M, _, _ = system.processBoundary(MM, {tag:lambda p: numpy.zeros((len(p),1))})
    
    MLU = ssl.splu(M)
    L = A = ssl.LinearOperator( M.shape, matvec=lambda x: MLU.solve(S* x), dtype=float)
    
    return ssl.eigen(L, k=n, which='SM', return_eigenvectors=False)

#    return sl.eigvals(S.todense(), M.todense())
    
    

@print_timing        
def laplacedirichlet(k, N, g, points):    
    tag = "B1"
    elements = H1Elements(k)
    quadrule = pyramidquadrature(k+1)
    system = SymmetricSystem(elements, quadrule, lambda m: buildcubemesh(N, m, tag), [tag])

    SM = system.systemMatrix(True)
    S, SIBs, Gs = system.processBoundary(SM, {tag:g})

    
    SG = SIBs[tag] * Gs[tag]
    print S.shape
    U = spsolve(S, -SG)[:,numpy.newaxis]

    return system.evaluate(points, U, Gs, False)

@print_timing
def poissondirichlet(k,N,g,f, points):
    tag = "B1"
    elements = H1Elements(k)
    quadrule = pyramidquadrature(k+1)
    system = SymmetricSystem(elements, quadrule, lambda m: buildcubemesh(N, m, tag), [tag])
    
    SM = system.systemMatrix(True)
    S, SIBs, Gs = system.processBoundary(SM, {tag:g})    
    SG = SIBs[tag] * Gs[tag]
    if f: 
        F = system.loadVector(f)
    else: F = 0
    
    t = Timer().start()
    U = spsolve(S, -F-SG)[:,numpy.newaxis]
    t.split("spsolve").show()
    
    return system.evaluate(points, U, Gs, False)

def poissonneumann(k,N,g,f, points):
    tag = "B1"
    elements = H1Elements(k)
    quadrule = pyramidquadrature(k+1)
    system = SymmetricSystem(elements, quadrule, lambda m: buildcubemesh(N,m,tag), [])
    SM = system.systemMatrix(True)
    S = SM[1:,:][:,1:] # the first basis fn is guaranteed to be associated with an external degree, so is a linear comb of all the others 
    F = 0 if f is None else system.loadVector(f)
    G = 0 if g is None else system.boundaryLoad({tag:g}, squarequadrature(k+1), trianglequadrature(k+1), False)    
    U = numpy.concatenate((numpy.array([0]), spsolve(S, G[tag][1:]-F[1:])))[:,numpy.newaxis]
    return system.evaluate(points, U, {}, False)    

def mixedpoissondual(k,N, g, f, points):
    import scipy.sparse as ss
    tag = "B1"
    hdiveltsA = HdivElements(k)
    hdiveltsB = HdivElements(k)
    hdiveltsBt = HdivElements(k)
    l2eltsB = L2Elements(k)
    l2eltsBt = L2Elements(k)
    quadrule = pyramidquadrature(k+1)
    meshevents = lambda m: buildcubemesh(N,m,tag)
    
    Asystem = SymmetricSystem(hdiveltsA, quadrule, meshevents, [])
    Bsystem = AsymmetricSystem(l2eltsB, hdiveltsB, quadrule, meshevents, [])
    Btsystem = AsymmetricSystem(hdiveltsBt, l2eltsBt, quadrule, meshevents, [])
    
    A = Asystem.systemMatrix(False)
    Bt = Btsystem.systemMatrix(True, False)
    B = Bsystem.systemMatrix(False, True)
    print A.shape, B.shape
    
    F = Bsystem.loadVector(f, False)
    gn = lambda x,n: g(x).reshape(-1,1,1) * n.reshape(-1,1,3)
    G = Btsystem.boundaryLoad({tag:gn}, squarequadrature(k+1), trianglequadrature(k+1), False)
    S = ss.bmat([[A, Bt],[B, None]])
    M = numpy.concatenate((G[tag], F))
    U = spsolve(S, M)[:,numpy.newaxis]
    return Btsystem.evaluate(points, U[len(G[tag]):], {}, False)    
    
    

def logxminus111(p):
    return (numpy.sum((p+numpy.array([1,1,1]))**2, axis=1)**(-1.0/2))[:,numpy.newaxis]
    
def tabulate():    
    kmax = 5
    nmax = 5
    points = uniformcubepoints(4)
    up = logxminus111(points)
    l2s = numpy.zeros((nmax,kmax))
    for k in range(1,kmax+1):
        for N in range(1,nmax+1):
            print k,N
            Up = laplacedirichlet(k,N, logxminus111, points)[:,numpy.newaxis]
            l2s[k-1,N-1] = math.sqrt(numpy.sum((Up - up)**2)/(len(points)))
        print l2s
 
def dopoisson():
    k = 5
    N = 3
    d = numpy.array([1,2,3])
    g = lambda p: numpy.sin(numpy.dot(p, d))[:,numpy.newaxis]
    f = lambda p: -sum(d**2) * g(p)
    points = uniformcubepoints(8)
    Up = poissondirichlet(k,N,g,f, points)[:,numpy.newaxis]
    up = g(points)
    print math.sqrt(numpy.sum((Up - up)**2)/(len(points)))

def dopoissonneumann():
    k = 4
    N = 3
    d = numpy.array([1,2,3])
    u = lambda p: numpy.sin(numpy.dot(p, d))[:,numpy.newaxis]
    gn = lambda p, n: (numpy.cos(numpy.dot(p,d))*numpy.dot(n,d))[:,numpy.newaxis]  
    f = lambda p: -sum(d**2) * u(p)
    points = uniformcubepoints(8)
    Up = poissonneumann(k,N,gn, f, points) [:,numpy.newaxis]
    up = u(points)
    l2 = math.sqrt(numpy.sum((Up - up)**2)/(len(points)))
    print l2

@print_timing
def dopoissonmixed():
    k = 3
    N = 3
    d = numpy.array([1,2,3])
    u = lambda p: numpy.sin(numpy.dot(p, d))[:,numpy.newaxis]
    f = lambda p: -sum(d**2) * u(p)
    points = uniformcubepoints(8)
    Up = mixedpoissondual(k,N, u, f, points) [:,numpy.newaxis]
    up = u(points)
    l2 = math.sqrt(numpy.sum((Up - up)**2)/(len(points)))
    print l2

        
if __name__ == "__main__":
    dopoissonmixed()
    #    tabulate()
    #    dopoissonneumann()
#    laplaceeigs(3,3)
#    dopoisson()
#    dopoissonmixed()