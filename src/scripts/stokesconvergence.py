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
    f = open("sc.final.dat", "w")
    f.write("k,N,e,ee,dofs\n")
    points, weights = pu.cubequadrature(12)
    up, ddup = pep.poisson(30, points[:,np.array([1,2])])
    l2 = lambda f: math.sqrt(np.sum(f.flatten() **2 * weights.flatten()))
    l2up = l2(up)
    print l2up
#    for N in range(2,15):
#        for k in range(1,10):
    for (k,N) in [(1,2),(2,2),(3,4),(4,2),(5,2),(6,2),(1,3),(2,3),(3,3),(4,3),(1,4),(2,4),(3,4),(1,5),(2,5),(3,5),(1,6),(2,6),(1,7),(2,7),(1,8),(1,9),(1,10),(1,11)]:
        meshevents = lambda m: pps.stokescubemesh(N, m)
    
        u, dofs = pps.stokespressure(k,meshevents,{pps.inputbdytag:pps.pfn(-0.5), pps.outputbdytag:pps.pfn(0.5)}, points, True,N==1)
        pt = points.transpose()
        ut = u.transpose()
        
        
        e = [l2(ut[0] - up)/l2up, l2(ut[1])/l2up, l2(ut[2])/l2up]
        ee = math.sqrt(e[0]**2 + e[1]**2, e[2]**2)
        print k, N, e, ee, dofs
        f.write("%s, %s, %s, %s\n"%(k,N,e,ee, dofs))
        f.flush()
#        if dofs > 30000: break
    f.close()
  

if __name__ == '__main__':
    convergence()
