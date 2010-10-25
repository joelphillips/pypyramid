'''
Created on Oct 25, 2010

@author: joel
'''
import pypyr.extra.poisson as pep
import pypyr.utils as pu
import pypyr.mesh as pm
import pypyr.physics.stokes as pps
import numpy as np
import math

def convergence():
    k = 2
    N = 2
    points, weights = pu.cubequadrature(8)
    meshevents = lambda m: pps.stokescubemesh(N, m)

    u, uu = pps.stokespressure(k,meshevents,{pps.inputbdytag:pps.pfn(0), pps.outputbdytag:pps.pfn(1.0)}, points)
    pt = points.transpose()
    ut = u.transpose()
    
    up, ddup = pep.poisson(60, points[:,np.array([1,2])])
    
    l2 = lambda f: math.sqrt(np.sum(f.flatten() **2 * weights.flatten()))
    e = [l2(ut[0] - up), l2(ut[1]), l2(ut[2])]
    print e
  

if __name__ == '__main__':
    convergence()