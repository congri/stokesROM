'''Script to load and plot solution to Stokes equation
'''

import dolfin as df
from matplotlib import pyplot as plt
import subprocess as sp
import numpy as np
import scipy.io as sio

# global params
porousMedium = 'circles'
meshes = np.arange(0, 4)  # vector of random meshes to load


# parameters for random field
volumeFraction = .5
nElements = 128
nMeshPolygon = 64
covarianceFunction = 'matern'
randFieldParams = [5.0]
lengthScale = [.008, .008]

#parameters for circular exclusions
nExclusions = 256
coordinateDistribution = 'uniform'
radiiDistribution = 'uniform'
r_params = (0.005, 0.03)
c_params = (.032, .968)


folderbase = '/home/constantin'
foldername = folderbase + '/python/data/stokesEquation/meshes/meshSize=' + str(nElements)
if porousMedium == 'randomField':
    foldername = foldername + '/randFieldDiscretization=' + str(nMeshPolygon) + '/cov=' + covarianceFunction +\
    '/params=' + str(randFieldParams) + '/l=' + str(lengthScale[0]) + '_' +\
    str(lengthScale[1]) + '/volfrac=' + str(volumeFraction)
elif porousMedium == 'circles':
    foldername = foldername + '/nCircExcl=' + str(nExclusions) + '/coordDist=' + coordinateDistribution +\
        '_c_params=' + str(c_params) + '/radiiDist=' + radiiDistribution + '_r_params=' + str(r_params)

for meshNumber in meshes:

    # load mesh from file
    try:
        mesh = df.Mesh(foldername + '/mesh' + str(meshNumber) + '.xml')

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
        uval = u.compute_vertex_values()
        pval = p.compute_vertex_values()
        sio.savemat('u_' + str(meshNumber) + '.mat', {'u': uval})
        sio.savemat('p_' + str(meshNumber) + '.mat', {'p': pval})

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.axes(ax1)
        df.plot(u, cmap=plt.cm.viridis, headwidth=0.005, headlength=0.005, scale=80.0, minlength=0.0001,
                width=0.0015, minshaft=0.01, headaxislength=0.1)
        # plot internal boundaries (boundary vertices
        bmesh = df.BoundaryMesh(mesh, 'exterior')
        xBoundary = bmesh.coordinates()
        plt.plot(xBoundary[:, 0], xBoundary[:, 1], 'ko', ms=.3, mew=0)
        plt.xticks([])
        plt.yticks([])
        ax1.set_title('vol. frac. = ' + str(volumeFraction) + ', flow u', fontsize=6)

        # plot pressure field
        plt.axes(ax2)
        pp = df.plot(p, cmap=plt.cm.inferno)
        pp = df.plot(p, cmap=plt.cm.inferno)
        pp = df.plot(p, cmap=plt.cm.inferno)
        cbar = plt.colorbar(pp, shrink=.318)
        cbar.ax.tick_params(labelsize=4, length=2, width=.2)
        plt.xticks([])
        plt.yticks([])
        ax2.set_title('pressure p', fontsize=6)

        # plot mesh
        plt.axes(ax3)
        df.plot(mesh, linewidth=0.05)
        plt.xticks([])
        plt.yticks([])
        ax3.set_title('FEM mesh', fontsize=6)



        figureFileName = './fig/stokes_vol=' + str(volumeFraction) + '_meshNr=' + str(meshNumber) + '.pdf'
        fig.savefig(figureFileName)
        sp.run(['pdfcrop', figureFileName, figureFileName])
        plt.close(fig)
    except:
        print('solution not found')

plt.show()