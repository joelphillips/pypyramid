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
    quads = []
    
    def addPyramid(self, pointids):
        pointids = np.array(pointids, dtype=object)
        self.triangles.extend([pointids[t] for t in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]])
        self.quads.extend([pointids[0:4]])

    def plot(self, fig):
        x,y,z = zip(*self.getPoints(np.array(self.triangles).flatten()))
        emm.triangular_mesh(x,y,z, np.arange(len(self.triangles)*3).reshape(-1,3), representation = 'wireframe', figure = fig, color=(0.5,0.5,0.5), opacity = 0.4, line_width=0.5)
        
    def plotbdy(self, fig, id):
        idset = set(map(tuple, self.boundaries[id]))
        bdyts = []
        for t in self.triangles:
            if idset.issuperset(set(t)): bdyts.append(t)
        for q in self.quads:
            if idset.issuperset(set(q)): 
                bdyts.append(q[[0,1,3]])
                bdyts.append(q[[2,1,3]])
            
        x,y,z = zip(*self.getPoints(np.array(bdyts).flatten()))
        emm.triangular_mesh(x,y,z, np.arange(len(bdyts)*3).reshape(-1,3), representation = 'surface', figure = fig, color=(0.5,0.2,0.5), opacity = 0.9, line_width=0.5)
        

        
if __name__ == "__main__":
    k = 2
    N = 2
    NP = 20
    points = pu.uniformcubepoints(NP)*0.6+0.2
    v = [[1,1,1]]
#    v = [[0,0,0]]
#    meshevents = lambda m: pps.stokescubemesh(N, m)
    OT = 'obstructiontag'
    meshevents = lambda m: pps.cubeobstruction(m, OT)
    mp = MeshPlotter()
    meshevents(mp)
    emm.figure(bgcolor=(1,1,1))

    
    u, p = pps.stokespressure(k,meshevents,{pps.inputbdytag:pps.pfn(0), pps.outputbdytag:pps.pfn(1.0)}, points, False, N==1)
    pt = points.reshape(NP,NP,NP,3).transpose((3,0,1,2))
    ut = u.reshape(NP,NP,NP,3).transpose((3,0,1,2))
    print p
        

#    emm.quiver3d(pt[0],pt[1],pt[2], ut[0],ut[1],ut[2])

    emm.flow(pt[0],pt[1],pt[2], ut[0],ut[1],ut[2], seedtype='plane', seed_resolution=10, seed_visible=True, scalars=p.reshape(NP,NP,NP))
    mp.plot(emm.gcf())
    mp.plotbdy(emm.gcf(), OT)
#    emm.figure()
#    uut = uu.transpose()
#    emm.quiver3d(pt[0],pt[1],pt[2], uut[0],uut[1],uut[2])
    emm.show()
