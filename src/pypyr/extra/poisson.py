'''
Created on Oct 25, 2010

@author: joel
'''
import pypyr.functions as pf
import pypyr.utils as pu
import scipy.linalg as sl
import math
import pylab

#import enthought.mayavi.mlab as emm

import numpy as np

def poisson(N, points):
    ''' Solves lap u = 1, on [0,1] with u=0 on boundary using a spectral element method and evaluates the solution at points 
    
        returns u(points), (lap u)(points)
    '''
    
    J = pf.getJacobi(1, 1)
    x,w = pu.legendrequadrature(N+3)
    xf = x.flatten()

    ns = np.arange(N+1,dtype=float)
    dpdp = np.diag((ns+1)**2 / (2*ns+3))    
    
    Px = J(N,xf) * xf * (1-xf)
    pp = np.dot(np.dot(Px, np.diag(w)), Px.transpose())   
    pp[(np.abs(pp) < 1E-15).nonzero()] = 0
    SS = (dpdp[:,np.newaxis,:,np.newaxis] * pp[np.newaxis,:,np.newaxis,:] + pp[:,np.newaxis,:,np.newaxis] * dpdp[np.newaxis,:,np.newaxis,:])
    S = SS.reshape((N+1)**2,(N+1)**2)
    
    F = np.zeros((N+1)**2)
    F[0] = 1.0/36 # (int_0^1 x * L_0)  
    X = sl.solve(S, -F, sym_pos=True)
    p0f = points[:,0].flatten()
    p1f = points[:,1].flatten()
    Jp0 = J(N, p0f)
    Jp1 = J(N, p1f)
    Ppoints0 = Jp0 * p0f * (1-p0f)
    Ppoints1 = Jp1 * p1f * (1-p1f)
    Ppts = (Ppoints0[:,np.newaxis,:]*Ppoints1[np.newaxis,:,:]).reshape((N+1)**2, len(points))
    
    ddfactor = -((ns+1) * (ns+2))[:,np.newaxis]
    ddPpoints0 = Jp0 * ddfactor  
    ddPpoints1 = Jp1 * ddfactor
    
    ddPpts = (ddPpoints0[:,np.newaxis,:] * Ppoints1[np.newaxis,:,:] + ddPpoints1[np.newaxis, :,:] * Ppoints0[:, np.newaxis, :]).reshape((N+1)**2, len(points))
    
    return np.dot(X, Ppts), np.dot(X,ddPpts)
    

def convergence(N):
    x,w= pu.squarequadrature(N)
    Nv = poisson(N,x)[0].flatten()
    e = np.zeros(N)
    for n in range(N):
        nv = poisson(n,x)[0].flatten()
        e[n] = math.sqrt(sum((Nv - nv)**2 * w))
    print e
    pylab.loglog(e)
    

if __name__ == '__main__':
    convergence(40)
    pylab.show()
#    
#    N = 6
#    NP = 50
##    points = np.vstack((np.arange(0,1,0.1),)*2).transpose() + np.array([[0.0001,0.2]])
#    points = pu.uniformsquarepoints(NP)
#    print points
#    u, ddu = poisson(N, points)
#    
#    emm.surf(points[:,0].reshape(NP,NP), points[:,1].reshape(NP,NP), u.reshape(NP,NP), warp_scale='auto')
#    emm.figure()
#    emm.surf(points[:,0].reshape(NP,NP), points[:,1].reshape(NP,NP), ddu.reshape(NP,NP), warp_scale='auto')
#    
#    emm.show()
    
    
    
