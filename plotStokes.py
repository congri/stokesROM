'''Script to load and plot solution to Stokes equation
'''

import dolfin as df
from matplotlib import pyplot as plt
import subprocess as sp


volumeFraction = .6
nElements = 128
nMeshPolygon = 64
covarianceFunction = 'matern'
randFieldParams = [5.0]
lengthScale = [.008, .008]

meshNumber = 2

folderbase = '/home/constantin/cluster'
foldername = folderbase + '/python/data/stokesEquation/meshes/meshSize=' + str(nElements) +\
    '/randFieldDiscretization=' + str(nMeshPolygon) + '/cov=' + covarianceFunction +\
    '/params=' + str(randFieldParams) + '/l=' + str(lengthScale[0]) + '_' +\
    str(lengthScale[1]) + '/volfrac=' + str(volumeFraction)

# load mesh from file
mesh = df.Mesh(foldername + '/mesh' + str(meshNumber) + '.xml')
df.plot(mesh)

# load solution from file
# Define mixed function space (Taylor-Hood)
u_e = df.VectorElement("CG", mesh.ufl_cell(), 2)
p_e = df.FiniteElement("CG", mesh.ufl_cell(), 1)
mixedEl = df.MixedElement([u_e, p_e])
W = df.FunctionSpace(mesh, mixedEl)
U = df.Function(W)
df.File(foldername + '/fullSolution' + str(meshNumber) + '.xml') >> U

# Get sub-functions
u, p = U.split()


fig, (ax1, ax2) = plt.subplots(1, 2)
plt.axes(ax1)
df.plot(u, cmap=plt.cm.viridis, headwidth=0.005, headlength=0.005, scale=80.0, minlength=0.0001,
        width=0.0015, minshaft=0.01, headaxislength=0.1)
# plot internal boundaries (boundary vertices
bmesh = df.BoundaryMesh(mesh, 'exterior')
xBoundary = bmesh.coordinates()
plt.plot(xBoundary[:, 0], xBoundary[:, 1], 'ko', ms=.4)
plt.xticks([])
plt.yticks([])
ax1.set_title('vol. frac. = ' + str(volumeFraction) + ', flow u')

# plot pressure field
plt.axes(ax2)
pp = df.plot(p, cmap=plt.cm.inferno)
pp = df.plot(p, cmap=plt.cm.inferno)
pp = df.plot(p, cmap=plt.cm.inferno)
plt.colorbar(pp, shrink=.5)
plt.xticks([])
plt.yticks([])
ax2.set_title('pressure p')


figureFileName = './fig/stokes_vol=' + str(volumeFraction) + '_meshNr=' + str(meshNumber) + '.pdf'
fig.savefig(figureFileName)
sp.run(['pdfcrop', figureFileName, figureFileName])

#plt.show()