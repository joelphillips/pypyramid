'''
Created on Oct 25, 2010

@author: joel
'''
import pypyr.utils as pu
import pypyr.mesh as pm
import pypyr.physics.stokes as pps
import pypyr.extra.poisson as pep
import numpy as np
import enthought.mayavi.mlab as emm
import matplotlib.pyplot as mpl


class MeshPlotter(pm.MeshBase):
    def __init__(self):
        pm.MeshBase.__init__(self)
        self.triangles = []
        self.quads = []
    
    def addPyramid(self, pointids):
        pointids = np.array(pointids, dtype=object)
        self.triangles.extend([pointids[t] for t in [[0,1,4],[1,2,4],[2,3,4],[3,0,4]]])
        self.quads.extend([pointids[0:4]])

    def plot(self, fig):
        x,y,z = zip(*self.getPoints(np.array(self.triangles).flatten()))
        emm.triangular_mesh(x,y,z, np.arange(len(self.triangles)*3).reshape(-1,3), representation = 'wireframe', figure = fig, color=(0.4,0.4,0.4), opacity = 0.4, line_width=0.5)
        
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
        emm.triangular_mesh(x,y,z, np.arange(len(bdyts)*3).reshape(-1,3), representation = 'surface', figure = fig, color=(0.2,0.2,0.2), opacity = 0.1, line_width=0.5)
        
def crosssection(x, NP, k, N, pN):
    points2 = pu.uniformsquarepoints(NP)
    points3 = np.hstack((np.ones((NP*NP,1)) * x, points2))
    meshevents = lambda m: pps.stokescubemesh(N, m)
    u, p = pps.stokespressure(k,meshevents,{pps.inputbdytag:pps.pfn(0), pps.outputbdytag:pps.pfn(1.0)}, points3, False, N==1)
    ur, _ = pep.poisson(pN, points2)
    pp = points2.reshape(NP,NP,2).transpose(2,0,1)
    mpl.figure(facecolor='white')
    c = mpl.contour(pp[0], pp[1], u[:,0].reshape(NP,NP), fontsize=14)
    mpl.clabel(c, inline=1, fontsize = 14)
    mpl.figure(facecolor='white')
    c = mpl.contourf(pp[0], pp[1], ur.reshape(NP,NP) - u[:,0].reshape(NP,NP), fontsize=14, colors = None, cmap=mpl.get_cmap('Greys'))
    mpl.colorbar(c, shrink=0.8)
#    mpl.clabel(c, inline=1, fontsize = 14)
    mpl.figure(facecolor='white')
    c = mpl.contourf(pp[0], pp[1], p.reshape(NP,NP), fontsize=14, colors = None, cmap=mpl.get_cmap('Greys'))
    mpl.colorbar(c, shrink=0.8)
#    mpl.clabel(c, inline=1, fontsize = 14)
#    mpl.show()
    

    
        
if __name__ == "__main__":
    k = 4
    N = 2
    NP = 12
    points = pu.uniformcubepoints(NP)
    v = [[1,1,1]]
#    v = [[0,0,0]]
    meshevents = lambda m: pps.stokescubemesh(N, m)
#    OT = 'obstructiontag'
#    meshevents = lambda m: pps.cubeobstruction(m, OT)
    mp = MeshPlotter()
    meshevents(mp)
    emm.figure(bgcolor=(1,1,1))

    u, p = pps.stokespressure(k,meshevents,{pps.inputbdytag:pps.pfn(0), pps.outputbdytag:pps.pfn(1.0)}, points, False, N==1)
    pt = points.reshape(NP,NP,NP,3).transpose((3,0,1,2))
    ut = u.reshape(NP,NP,NP,3).transpose((3,0,1,2))        

    emm.quiver3d(pt[0],pt[1],pt[2], ut[0],ut[1],ut[2], opacity = 0.4)

#    emm.flow(pt[0],pt[1],pt[2], ut[0],ut[1],ut[2], seedtype='plane', seed_resolution=10, seed_visible=True, scalars=p.reshape(NP,NP,NP))
    mp.plot(emm.gcf())
    mp.plotbdy(emm.gcf(), pps.closedbdytag)
    
#    emm.figure()
#    uut = uu.transpose()
#    emm.quiver3d(pt[0],pt[1],pt[2], uut[0],uut[1],uut[2])
    emm.show()
