'''
Created on Aug 17, 2010

@author: joel
'''

from pypyr.mappings import derham3dweights, psijac, psiinvjac, psidet,  psi
from pypyr.diffforms import DiffForm, CatDiffForm, MapDiffForm, derham
from pypyr.functions import QSpace, ZeroFns

def buildRForms(k):
    w = derham3dweights(psijac, psiinvjac, psidet)
    R0Q = QSpace(k,k,k)
    R0I = DiffForm([R0Q], derham)
    R0 = MapDiffForm(R0I, psi, w)
    
    R1Q1 = QSpace(k-1,k,k+1)
    R1Q10 = ZeroFns(R1Q1.nfns)
    R1Q2 = QSpace(k,k-1,k+1)
    R1Q20 = ZeroFns(R1Q2.nfns)
    R1Ia = CatDiffForm([DiffForm([R1Q1,R1Q10,R1Q10], derham[1:]), DiffForm([R1Q20,R1Q2,R1Q20], derham[1:])])
    R1I = CatDiffForm([R1Ia, R0I.D()])
    R1 = MapDiffForm(R1I, psi, w[1:]) 
    
    R2Q = QSpace(k-1,k-1,k+2)
    R2Q0 = ZeroFns(R2Q.nfns)
    R2Ia = DiffForm([R2Q0, R2Q0, R2Q], derham[2:])
    R2I = CatDiffForm([R2Ia, R1Ia.D()])
    R2 = MapDiffForm(R2I, psi, w[2:])
    
    R3 = MapDiffForm(DiffForm([QSpace(k-1,k-1,k+3)], []), psi, w[3:])
    return [R0,R1,R2,R3]

                                    