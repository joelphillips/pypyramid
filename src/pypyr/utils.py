'''
Created on Aug 17, 2010

@author: joel
'''
import scipy.special.orthogonal
import numpy

def pyrpoints(n):
    p = []
    for z in numpy.arange(n+1):
        for y in numpy.arange(n+1-z):
            for x in numpy.arange(n+1-z):
                p.append([x,y,z])
    return numpy.array(p,dtype=numpy.float64)/n

def polyn(c, p):
    return numpy.product([p[:,i]**c[i] for i in [0,1,2]], axis = 0)

def pyramidquadrature(n):
    x00,w00 = scipy.special.orthogonal.p_roots(n)
    x02,w02 = scipy.special.orthogonal.j_roots(n,2,0)
    x00s = (x00+1)/2
    x02s = (x02+1)/2
    
    g = numpy.mgrid[0:n,0:n,0:n].reshape(3,-1)
    w = w00[g[0]] * w00[g[1]] * w02[g[2]] / 32 # a factor of 2 for the legendres and 8 for the jacobi20
    z = x02s[g[2]]
    x = numpy.vstack((x00s[g[0]]*(1-z), x00s[g[1]]* (1-z), z))

    return x.transpose(), w
    
    
def trianglequadrature(n):
    """ Degree n quadrature points and weights on a triangle (0,0)-(1,0)-(0,1)"""

    x00,w00 = scipy.special.orthogonal.p_roots(n)
    x01,w01 = scipy.special.orthogonal.j_roots(n,1,0)
    x00s = (x00+1)/2
    x01s = (x01+1)/2
    w = numpy.outer(w01, w00).reshape(-1,1) / 8 # a factor of 2 for the legendres and 4 for the jacobi10
    x = numpy.outer(x01s, numpy.ones(x00s.shape)).reshape(-1,1)
    y = numpy.outer(1-x01s, x00s).reshape(-1,1)
    return numpy.hstack((x, y)), w

def squarequadrature(n):
    x00,w00 = legendrequadrature(n)
    g = numpy.mgrid[0:n,0:n].reshape(2,-1)
    w = w00[g[0]] * w00[g[1]]
    x = numpy.hstack((x00[g[0]], x00[g[1]]))
    return x, w

def cubequadrature(n):
    x00,w00 = legendrequadrature(n)
    g = numpy.mgrid[0:n,0:n,0:n].reshape(3,-1)
    w = w00[g[0]] * w00[g[1]] * w00[g[2]]
    x = numpy.hstack((x00[g[0]], x00[g[1]], x00[g[2]]))
    return x, w


def legendrequadrature(n):
    """ Legendre quadrature points on [0,1] """
    x00,w00 = scipy.special.orthogonal.p_roots(n)
    return (x00.reshape(-1,1)+1)/2, w00/2

def uniformsquarepoints(np):
    return numpy.linspace(0,1,np)[numpy.mgrid[0:np,0:np].reshape(2,-1)].transpose()

def uniformcubepoints(np):
    return numpy.linspace(0,1,np)[numpy.mgrid[0:np,0:np,0:np].reshape(3,-1)].transpose()
