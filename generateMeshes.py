'''This is a script to generate and save meshes'''

import scipy.stats as stats
from randomFieldGeneration import RandomField as rf
import porousMedia as pm
import dolfin as df
import os
import numpy as np
import scipy.io as sio
import time


# Global parameters
mode = 'nonOverlappingCircles'
nMeshes = 1
nElements = 128  # PDE discretization
foldername1 = '/home/constantin/python/data/stokesEquation/meshes/meshSize=' + str(nElements)


#Parameters only for 'circles' mode
nExclusions = 512
coordinateDist = 'uniform'
c_params = (.022, .978)  # to avoid circles on boundaries
radiiDist = 'uniform'
r_params = (.005, .025)


#parameters only for 'randomField' mode
volumeFraction = .9  # Volume fraction of voids
lengthScale = [.004, .004]
nMeshPolygon = 128  # image discretization of random material; needed for polygones
# redUnitSq or marchingSquares - one is faster, the other finer
# ATTENTION: Using redUnitSq, the PETSc solver for the stokes problem does not converge
modeRF = 'marchingSquares'

if mode == 'randomField':

    print('Generate meshes from discretized random fields...')
    cutoff = stats.norm.ppf(volumeFraction)
    print('cutoff = ', cutoff)
    print('Random field mesh mode mode: ', modeRF)


    def potential(x):
        # Potential to avoid blobs on boundary
        p = 0
        # Penalty for boundaries at 0
        sigma = 1e-4
        prefactor = 3e-4
        p -= prefactor * stats.laplace.pdf(x[0], 0, sigma)
        p -= prefactor * stats.laplace.pdf(x[1], 0, sigma)
        p -= prefactor * stats.laplace.pdf(x[0], 1, sigma)
        p -= prefactor * stats.laplace.pdf(x[1], 1, sigma)
        return p


    def addfunctions(f, g):
        # add functions f and g
        def h(x):
            return f(x) + g(x)

        return h


    randomFieldObj = rf()
    randomFieldObj.lengthScale = lengthScale

    if modeRF == 'redUnitSq':
        nMeshPolygon = modeRF   # for file path
    foldername = foldername1 + '/randFieldDiscretization=' + str(nMeshPolygon) + '/cov=' + \
                 randomFieldObj.covarianceFunction + '/params=' + str(randomFieldObj.params) + '/l=' + \
                 str(randomFieldObj.lengthScale[0]) + '_' + str(randomFieldObj.lengthScale[1]) + \
                 '/volfrac=' + str(volumeFraction)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for i in range(0, nMeshes):
        randomField = randomFieldObj.sample()
        randomField = addfunctions(randomField, potential)


        if modeRF == 'marchingSquares':
            discretizedRandomField = pm.discretizeRandField(randomField,
                                                            nDiscretize=(nMeshPolygon, nMeshPolygon))

            contours = pm.findPolygones(discretizedRandomField, cutoff)
            contours = pm.rescalePolygones(contours, nDiscretize=(nMeshPolygon, nMeshPolygon))
            domain = pm.substractPolygones(contours)

            # Generate mesh - this step is expensive
            mesh = pm.generateMesh(domain)
        elif modeRF == 'redUnitSq':
            mesh = df.UnitSquareMesh(nElements, nElements, 'right/left')

            cells = mesh.cells()
            vtx_coord = mesh.coordinates()

            class RandField(df.SubDomain):
                def inside(self, x, on_boundary):
                    return randomField(x) > cutoff

            rff = RandField()
            subdomains = df.CellFunction('size_t', mesh)
            subdomains.set_all(0)
            rff.mark(subdomains, 1)

            marked_cells = df.SubsetIterator(subdomains, 1)

            cell_to_delete = []
            for cell in marked_cells:
                cell_to_delete.append(cell.index())

            print('cells to delete: ', cell_to_delete)
            cells = np.delete(cells, cell_to_delete, axis=0)

            mesh = df.Mesh()
            editor = df.MeshEditor()
            editor.open(mesh, 2, 2)
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

            editor.close()

        # save mesh in xml format for later use
        filename = foldername + '/mesh' + str(i) + '.xml'
        mesh_file = df.File(filename)
        mesh_file << mesh

        # save vertex coordinates and cell connectivity to mat file for easy read-in to matlab
        sio.savemat(foldername + '/mesh' + str(i) + '.mat',
                    {'x': mesh.coordinates(), 'cells': mesh.cells() + 1.0})


elif mode == 'circles':
    print('Generating mesh with circular exclusions...')

    foldername = foldername1 + '/nCircExcl=' + str(nExclusions) + '/coordDist=' +\
                 coordinateDist + '_c_params=' + str(c_params) +\
                 '/radiiDist=' + radiiDist + '_r_params=' + str(r_params)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for i in range(0, nMeshes):
        if coordinateDist == 'uniform':
            exclusionCenters = (c_params[1] - c_params[0])*np.random.rand(nExclusions, 2) + c_params[0]

        if radiiDist == 'uniform':
            exclusionRadii = (r_params[1] - r_params[0]) * np.random.rand(nExclusions) + r_params[0]
        elif radiiDist == 'exponential':
            exclusionRadii = np.random.exponential(r_params, nExclusions)

        domain = pm.substractCircles(exclusionCenters, exclusionRadii)

        # Generate mesh - this step is expensive
        mesh = pm.generateMesh(domain)

        #save mesh in xml format for later use
        filename = foldername + '/mesh' + str(i) + '.xml'
        mesh_file = df.File(filename)
        mesh_file << mesh

        #save vertex coordinates and cell connectivity to mat file for easy read-in to matlab
        sio.savemat(foldername + '/mesh' + str(i) + '.mat',
                    {'x': mesh.coordinates(), 'cells': mesh.cells() + 1.0})


elif mode == 'nonOverlappingCircles':
    print('Generating mesh with non-overlapping circular exclusions...')

    foldername = foldername1 + '/nNonOverlapCircExcl=' + str(nExclusions) + '/coordDist=' +\
                 coordinateDist + '_c_params=' + str(c_params) +\
                 '/radiiDist=' + radiiDist + '_r_params=' + str(r_params)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for i in range(0, nMeshes):

        exclusionCenters = np.empty((0, 2))
        exclusionRadii = np.empty((0, 1))
        currentExclusions = 0
        t_start = time.time()
        t_elapsed = 0
        t_lim = 10.0
        while currentExclusions < nExclusions and t_elapsed < t_lim:
            if coordinateDist == 'uniform':
                exclusionCenter = (c_params[1] - c_params[0])*np.random.rand(1, 2) + c_params[0]

            if radiiDist == 'uniform':
                exclusionRadius = (r_params[1] - r_params[0]) * np.random.rand(1) + r_params[0]
            elif radiiDist == 'exponential':
                exclusionRadius = np.random.exponential(r_params, 1)

            overlap = False
            iter = 0
            for x_circ in exclusionCenters:
                dist = np.linalg.norm(x_circ - exclusionCenter)
                if dist <= exclusionRadius + exclusionRadii[iter]:
                    #disks overlap
                    overlap = True
                    break
                iter += 1

            if not overlap:
                exclusionCenters = np.append(exclusionCenters, exclusionCenter, axis=0)
                exclusionRadii = np.append(exclusionRadii, exclusionRadius)
                currentExclusions += 1


        domain = pm.substractCircles(exclusionCenters, exclusionRadii)

        # Generate mesh - this step is expensive
        mesh = pm.generateMesh(domain)

        #save mesh in xml format for later use
        filename = foldername + '/mesh' + str(i) + '.xml'
        mesh_file = df.File(filename)
        mesh_file << mesh

        #save vertex coordinates and cell connectivity to mat file for easy read-in to matlab
        sio.savemat(foldername + '/mesh' + str(i) + '.mat',
                    {'x': mesh.coordinates(), 'cells': mesh.cells() + 1.0})
