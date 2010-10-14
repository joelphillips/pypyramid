'''
Created on Oct 13, 2010

@author: joel
'''
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import scipy.sparse.linalg.dsolve as ssld

import numpy as np
import math
from pypyr.timing import print_timing

@print_timing
def directsolve(A,B,f,g):
    S = ss.bmat([[A, B.transpose()],[B, None]])
    M = np.concatenate((f.flatten(), g.flatten()))
    U = ssl.spsolve(S, M)
    return U[:len(f)], U[len(f):]

@print_timing
def uzawa(A, B, f, g):
    Asolve = ssld.factorized(A)
    p = np.zeros(B.shape[0])
    r = 0.0001
    while True:
        u = Asolve(f.flatten() - B.transpose() * p)
        pd = B * u - g.flatten()
        p += r * pd
        pdl2 = math.sqrt(np.sum(pd * pd))
        print pdl2
        if pdl2 < 1E-6: break
    return u, p

@print_timing
def mixedcg(A, B, f, g):
    Asolve = ssld.factorized(A)
    Aif = Asolve(f.flatten())
    BAf = B * Aif
    
    def BAiBt(p):
        return B * Asolve(B.transpose() * p)
    
    p, info = ssl.bicgstab(ssl.LinearOperator((B.shape[0],)*2, matvec = BAiBt, dtype=float), BAf - g.flatten())
    print "bicgstab result ",info
    u = Aif - Asolve(B.transpose() * p)
    return u,p
