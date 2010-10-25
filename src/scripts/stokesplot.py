'''
Created on Oct 25, 2010

@author: joel
'''
import pypyr.utils as pu
import pypyr.mesh as pm
import pypyr.physics.stokes as pps
import numpy as np
import enthought.mayavi.mlab as emm


class MeshPlotter(pm.MeshBase):
    triangles = []
    def addPyramid(self, pointids):
        pointids = np.array(pointids, dtype=object)
        self.triangles.extend([pointids[t] for t in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]])
    
    def plot(self, fig):
        x,y,z = zip(*self.getPoints(np.array(self.triangles).flatten()))
        emm.triangular_mesh(x,y,z, np.arange(len(self.triangles)*3).reshape(-1,3), representation = 'wireframe', figure = fig, color=(0,0,0), line_width=0.5)
        
  
        
if __name__ == "__main__":
    k = 3
    N = 2
    points = pu.uniformcubepoints(8)
    v = [[1,1,1]]
#    v = [[0,0,0]]
    meshevents = lambda m: pps.stokescubemesh(N, m)
    mp = MeshPlotter()
    meshevents(mp)
    u, uu = pps.stokespressure(k,meshevents,{pps.inputbdytag:pps.pfn(0), pps.outputbdytag:pps.pfn(1.0)}, points)
    pt = points.transpose()
    ut = u.transpose()
        
    emm.figure(bgcolor=(1,1,1))

    emm.quiver3d(pt[0],pt[1],pt[2], ut[0],ut[1],ut[2])

#    emm.flow(pt[0],pt[1],pt[2], ut[0],ut[1],ut[2])
    mp.plot(emm.gcf())
#    emm.figure()
#    uut = uu.transpose()
#    emm.quiver3d(pt[0],pt[1],pt[2], uut[0],uut[1],uut[2])
    emm.show()
