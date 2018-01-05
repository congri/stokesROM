import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from randomFieldGeneration import RandomField as rf


mesh = df.UnitSquareMesh(8, 8, 'right/left')


cells = mesh.cells()
vtx_coord = mesh.coordinates()


class CenterSq(df.SubDomain):
    def inside(self, x, on_boundary):
        return 0.3 < x[0] < .7 and 0.3 < x[1] < .7


randomFieldObj = rf()
randomField = randomFieldObj.sample()


class RandField(df.SubDomain):
    def inside(self, x, on_boundary):
        return randomField(x) > 0


rff = CenterSq()
subdomains = df.CellFunction('size_t', mesh)
subdomains.set_all(0)
rff.mark(subdomains, 1)

df.plot(subdomains)
fig = plt.figure()

marked_cells = df.SubsetIterator(subdomains, 1)

cell_to_delete = []
for cell in marked_cells:
    cell_to_delete.append(cell.index())

print('cells to delete: ', cell_to_delete)



cells = np.delete(cells, cell_to_delete, axis=0)

new_mesh = df.Mesh()
editor = df.MeshEditor()
editor.open(new_mesh, 2, 2)
editor.init_vertices(len(vtx_coord))
editor.init_cells(len(cells))

vert_id = 0
for vert in vtx_coord:
    editor.add_vertex(vert_id, vert)
    vert_id += 1

cell_id = 0
for c in range(len(cells)):
    editor.add_cell(cell_id, cells[c][0], cells[c][1], cells[c][2])
    cell_id += 1 

print('cell_id: ', cell_id)
print('cell connectivity: ', cells)

print('vert_id: ', vert_id)
print('vertex coordinates: ', vtx_coord)

editor.close()
df.plot(new_mesh)
plt.show()
