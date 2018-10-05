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
nMeshes = 8
nElements = 128  # PDE discretization
foldername1 = '/home/constantin/python/data/stokesEquation/meshSize=' + str(nElements)


#Parameters only for 'circles' mode
nExclusionsDist='logn'
nExclusionParams = (5.6, .6)
coordinateDist = 'gauss'
# to avoid circles on boundaries. Min. distance of circle centers to (lo., r., u., le.) boundary
# negative margin means no margin
margins = (0.01, .01, .01, .01)
origin_margin = .005
substractCorners = False     #Substracts circles from domain corners s.t. flow cannot pass
radiiDist = 'logn'
r_params = (-4.5, .15)
coordinate_cov = [[0.55, -0.45], [-0.45, 0.55]]
coordinate_mu = [.8, .8]


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

    foldername = foldername1 + '/nCircExcl=' + str(nExclusionParams[0]) + '-' + str(nExclusionParams[1]) + '/coordDist=' +\
                 coordinateDist + '_margins=' + str(margins) + '/radiiDist=' + radiiDist + '_r_params=' + str(r_params)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for i in range(0, nMeshes):
        nExclusions = np.random.randint(nExclusionParams[0], nExclusionParams[1])
        print('nExclusions = ', nExclusions)
        if coordinateDist == 'uniform':
            exclusionCentersX = (1.0 - margins[1] - margins[3])*np.random.rand(nExclusions, 1) + margins[3]
            exclusionCentersY = (1.0 - margins[0] - margins[2])*np.random.rand(nExclusions, 1) + margins[0]
            exclusionCenters = np.concatenate((exclusionCentersX, exclusionCentersY), axis=1)
        if radiiDist == 'uniform':
            exclusionRadii = (r_params[1] - r_params[0]) * np.random.rand(nExclusions) + r_params[0]
        elif radiiDist == 'exponential':
            exclusionRadii = np.random.exponential(r_params, nExclusions)
        elif radiiDist == 'logn':
            # log normal radii distribution
            exclusionRadii = np.random.lognormal(r_params[0], r_params[1], nExclusions)

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

    foldername = foldername1 + '/nonOverlappingDisks/margins=' + str(margins[0]) + '_' + str(margins[1]) + '_' + \
                 str(margins[2]) + '_' + str(margins[3])

    if nExclusionsDist == 'uniform':
        # outdated!!
        foldername = foldername1 + '/nNonOverlapCircExcl=' + str(nExclusionParams[0]) + '-' + str(nExclusionParams[1]) +\
            '/coordDist=' + coordinateDist
    elif nExclusionsDist == 'logn':

        foldername += '/N~logn/mu=' + str(nExclusionParams[0]) + '/sigma=' + str(nExclusionParams[1]) + \
                     '/x~' + coordinateDist

    if coordinateDist == 'gauss':
        foldername += '/mu=' + str(coordinate_mu[0]) + '_' + str(coordinate_mu[1]) + '/cov=' + \
                      str(coordinate_cov[0][0]) + '_' + str(coordinate_cov[0][1]) + '_' + str(coordinate_cov[1][1]) +\
                      '/'
    elif coordinateDist == 'gauss_randmu':
        foldername += '/mu=rand' + '/cov=' + \
                      str(coordinate_cov[0][0]) + '_' + str(coordinate_cov[0][1]) + '_' + str(coordinate_cov[1][1]) + \
                      '/'

    foldername += '/r~' + radiiDist + '/mu=' + str(r_params[0]) + '/sigma=' + str(r_params[1])
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    mesh_name_iter = 0
    while mesh_name_iter < nMeshes:
        if nExclusionsDist == 'uniform':
            nExclusions = np.random.randint(nExclusionParams[0], nExclusionParams[1])
        elif nExclusionsDist == 'logn':
            nExclusions = np.random.lognormal(nExclusionParams[0], nExclusionParams[1])
        nExclusions_upper_bound = 30000         # this is needed to avoid segfaults in mesh generation
        nExclusions = min(nExclusions, nExclusions_upper_bound)
        print('nExclusions = ', nExclusions)
        exclusionCenters = np.empty([0, 2])
        exclusionRadii = np.empty([0, 1])
        if substractCorners:
            exclusionCenters = np.append(exclusionCenters, np.array([[0.0, 0.0]]), axis=0)
            exclusionCenters = np.append(exclusionCenters, np.array([[1.0, 0.0]]), axis=0)
            exclusionCenters = np.append(exclusionCenters, np.array([[0.0, 1.0]]), axis=0)
            exclusionCenters = np.append(exclusionCenters, np.array([[1.0, 1.0]]), axis=0)
            exclusionRadii = np.append(exclusionRadii, 0.03)
            exclusionRadii = np.append(exclusionRadii, 0.03)
            exclusionRadii = np.append(exclusionRadii, 0.03)
            exclusionRadii = np.append(exclusionRadii, 0.03)

        currentExclusions = 0
        t_start = time.time()
        t_elapsed = 0
        t_lim = 3600.0
        if coordinateDist == 'gauss_randmu':
            coordinate_mu = np.random.rand(2)

        while currentExclusions < nExclusions and t_elapsed < t_lim:
            if coordinateDist == 'uniform':
                exclusionCenter = np.random.rand(1, 2)
            elif coordinateDist == 'gauss' or coordinateDist == 'gauss_randmu':
                exclusionCenter = np.empty([1, 2])
                exclusionCenter[0] = np.random.multivariate_normal(coordinate_mu, coordinate_cov)

            if radiiDist == 'uniform':
                exclusionRadius = (r_params[1] - r_params[0]) * np.random.rand(1) + r_params[0]
            elif radiiDist == 'exponential':
                exclusionRadius = np.random.exponential(r_params, 1)
            elif radiiDist == 'logn':
                exclusionRadius = np.random.lognormal(r_params[0], r_params[1])

            # check for overlap with other disks

            # old version, inefficient
            # overlap = False
            # iter = 0
            # for x_circ in exclusionCenters:
            #     dist = np.linalg.norm(x_circ - exclusionCenter)
            #     if dist <= exclusionRadius + exclusionRadii[iter]:
            #         # disks overlap
            #         overlap = True
            #         break
            #     iter += 1

            overlap = np.any((exclusionRadius + exclusionRadii) >=
                             np.linalg.norm(exclusionCenter - exclusionCenters, axis=1))

            if not overlap:
                # check for overlap with domain boundary
                onBoundary = False
                if (((exclusionCenter[0, 1] - exclusionRadius) < margins[0]) and margins[0] >= 0) or \
                    (((exclusionCenter[0, 0] + exclusionRadius) > (1 - margins[1])) and margins[1] >= 0) or \
                    (((exclusionCenter[0, 1] + exclusionRadius) > (1 - margins[2])) and margins[2] >= 0) or \
                        (((exclusionCenter[0, 0] - exclusionRadius) < margins[3]) and margins[3] >= 0) or \
                        ((np.linalg.norm(exclusionCenter) - exclusionRadius) < origin_margin):
                    onBoundary = True

                if not onBoundary:
                    # check if disk is out of domain
                    outOfDomain = False
                    if (exclusionCenter[0, 1] + exclusionRadius) < 0 or \
                        (exclusionCenter[0, 0] - exclusionRadius) > 1 or \
                        (exclusionCenter[0, 1] - exclusionRadius) > 1 or \
                            (exclusionCenter[0, 0] + exclusionRadius) < 0:
                        outOfDomain = True

                    if not outOfDomain:
                        exclusionCenters = np.append(exclusionCenters, exclusionCenter, axis=0)
                        exclusionRadii = np.append(exclusionRadii, exclusionRadius)
                        currentExclusions += 1
            t_elapsed = time.time() - t_start
        print('Non-overlapping disks drawn.')
        print('real number of exclusions == ', exclusionRadii.size)
        print('Smallest radius == ', np.amin(exclusionRadii))

        domain = pm.substractCircles(exclusionCenters, exclusionRadii)

        # Generate mesh - this step is expensive
        print('Disks substracted.')
        try:
            mesh = pm.generateMesh(domain, nElements)
            print('mesh generated.')
            # save mesh in xml format for later use
            # check how many meshes already exist and name the mesh accordingly (allows parallel mesh generation jobs)
            mesh_name_iter = 0
            while os.path.isfile(foldername + '/mesh' + str(mesh_name_iter) + '.xml'):
                mesh_name_iter += 1

            filename = foldername + '/mesh' + str(mesh_name_iter) + '.xml'
            mesh_file = df.File(filename)
            mesh_file << mesh

            # save vertex coordinates and cell connectivity to mat file for easy read-in to matlab
            sio.savemat(foldername + '/mesh' + str(mesh_name_iter) + '.mat',
                        {'x': mesh.coordinates(), 'cells': mesh.cells() + 1.0})
            # save microstructural information, i.e. centers and radii of disks
            sio.savemat(foldername + '/microstructureInformation' + str(mesh_name_iter) + '.mat',
                        {'diskCenters': exclusionCenters, 'diskRadii': exclusionRadii})
        except:
            print('Mesh generation failed. Trying again...')

