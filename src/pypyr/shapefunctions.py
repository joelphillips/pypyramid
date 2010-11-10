'''
The Pyramidal approximation spaces.  See the paper "Numerical integration for high order pyramidal elements"

For each space, we build first build the Helmholtz decomposition, then map to the finite pyramid.

Created on Aug 17, 2010

@author: joel
'''

from pypyr.mappings import derham3dweights, psijac, psiinvjac, psidet,  psi
from pypyr.diffforms import DiffForm, CatDiffForm, MapDiffForm, derham
from pypyr.functions import QSpace, ZeroFns

w = derham3dweights(psijac, psiinvjac, psidet)


def R1GradFreeI(k):
    R1Q1 = QSpace(k-1,k,k+1)
    R1Q10 = ZeroFns(R1Q1.nfns)
    R1Q2 = QSpace(k,k-1,k+1)
    R1Q20 = ZeroFns(R1Q2.nfns)
    return CatDiffForm([DiffForm([R1Q1,R1Q10,R1Q10], derham[1:]), DiffForm([R1Q20,R1Q2,R1Q20], derham[1:])])
    
def R2CurlFreeI(k):
    R2Q = QSpace(k-1,k-1,k+2)
    R2Q0 = ZeroFns(R2Q.nfns)
    return DiffForm([R2Q0, R2Q0, R2Q], derham[2:])

def R0Forms(k):
    R0Q = QSpace(k,k,k)
    R0I = DiffForm([R0Q], derham)
    return MapDiffForm(R0I, psi, w)

def R1Forms(k):
    R0D = DiffForm([QSpace(k,k,k,1)], derham).D()
    R1I = CatDiffForm([R1GradFreeI(k), R0D])
    return MapDiffForm(R1I, psi, w[1:]) 

def R2Forms(k):
    R2I = CatDiffForm([R2CurlFreeI(k), R1GradFreeI(k).D()])
    return MapDiffForm(R2I, psi, w[2:])
    
def R3Forms(k):
    return MapDiffForm(DiffForm([QSpace(k-1,k-1,k+3)], []), psi, w[3:])

def R2FormsDivFree(k):
    return MapDiffForm(R1GradFreeI(k).D(), psi, w[2:])

                                    