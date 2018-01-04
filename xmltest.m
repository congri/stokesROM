%test file to read in xml data

xmlmesh = xmlread('/home/constantin/python/data/stokesEquation/meshes/meshSize=128/nCircExcl=256/coordDist=uniform/radiiDist=uniform_r_params=(0.005, 0.03)/mesh0.xml');


dolfin = item(xmlmesh, 0)
mesh = item(dolfin, 1)
cells = item(mesh, 3)
cell1 = item(cells, 1)
cell2 = item(cells, 3)


