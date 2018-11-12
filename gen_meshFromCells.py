'''Generate mesh from a list of vertices and cells only'''
import scipy.io as sio
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

foldername = '/home/constantin/cluster/python/data/stokesEquation/meshSize=256/nonOverlappingDisks/' \
             'margins=0.003_0.003_0.003_0.003/N~logn/mu=8.1/sigma=0.6/x~tiles/r~logn/mu=-5.53/sigma=0.3'
mesh_name_iter = 2
mesh_data = sio.loadmat(foldername + '/mesh' + str(mesh_name_iter) + '.mat')
x = mesh_data['x']
cells = mesh_data['cells']
cells -= 1
cells = np.array(cells, dtype=np.uintp)

# np.savetxt(foldername + '/mesh' + str(mesh_name_iter), cells, fmt='%d')

editor = df.MeshEditor()
mesh = df.Mesh()
editor.open(mesh, "triangle", 2, 2)
editor.init_vertices(x.shape[0])
editor.init_cells(cells.shape[0])
for k, point in enumerate(x):
    editor.add_vertex(k, point[:2])
for k, cell in enumerate(cells):
    editor.add_cell(k, cell)
editor.close()


# df.plot(mesh)
# plt.show()


