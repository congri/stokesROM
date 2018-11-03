'''This is a script to generate and save meshes'''

import scipy.stats as stats
from randomFieldGeneration import RandomField as rf
import porousMedia as pm
import dolfin as df
import os
from shutil import copyfile
import numpy as np
import scipy.io as sio
import time


# Global parameters
mode = 'nonOverlappingCircles'
load_microstructures = True         #if microstructural data has been generated elsewhere
nMeshes = 2048
nElements = 256  # PDE discretization
foldername1 = '/home/constantin/python/data/stokesEquation/meshSize=' + str(nElements)


#Parameters only for 'circles' mode
nExclusionsDist='logn'
nExclusionParams = (8.35, .6)
coordinateDist = 'engineered'
# to avoid circles on boundaries. Min. distance of circle centers to (lo., r., u., le.) boundary
# negative margin means no margin
margins = (0.003, .003, .003, .003)
origin_margin = .005
substractCorners = False     #Substracts circles from domain corners s.t. flow cannot pass
radiiDist = 'logn'
r_params = (-5.53, .3)
# for x~gauss
coordinate_cov = [[0.55, -0.45], [-0.45, 0.55]]
coordinate_mu = [.8, .8]
# for x~GP
covFun = 'squaredExponential'
cov_l = 0.1
sig_scale = 1.5


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
            diskCenters = np.concatenate((exclusionCentersX, exclusionCentersY), axis=1)
        if radiiDist == 'uniform':
            diskRadii = (r_params[1] - r_params[0]) * np.random.rand(nExclusions) + r_params[0]
        elif radiiDist == 'exponential':
            diskRadii = np.random.exponential(r_params, nExclusions)
        elif radiiDist == 'logn':
            # log normal radii distribution
            diskRadii = np.random.lognormal(r_params[0], r_params[1], nExclusions)

        domain = pm.substractCircles(diskCenters, diskRadii)

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
    elif coordinateDist == 'GP':
        foldername += '/cov=' + covFun + '/l=' + str(cov_l) + '/sig_scale=' + str(sig_scale) + '/'

    elif coordinateDist == 'engineered' or coordinateDist == 'tiles':
        foldername += '/'

    foldername += 'r~' + radiiDist + '/mu=' + str(r_params[0]) + '/sigma=' + str(r_params[1])
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    if load_microstructures:
        # first copy 'microstructureInformation_nomesh' to 'microstructureInformation', to give signal that mesh is
        #  generated, so that no other job is taking the same microstructure to generate a mesh

        mesh_name_iter = 0
        # copy microstructureInformation file, this is the signal that a job is already generating a mesh
        while mesh_name_iter < nMeshes:

            # mesh_name iter is the first mesh that has not yet been and is not currently generated
            mesh_name_iter = 0
            while os.path.isfile(foldername + '/microstructureInformation' + str(mesh_name_iter) + '.mat'):
                mesh_name_iter += 1
            print('mesh_name_iter == ', mesh_name_iter)

            copyfile(foldername + '/microstructureInformation_nomesh' + str(mesh_name_iter) + '.mat',
                     foldername + '/microstructureInformation' + str(mesh_name_iter) + '.mat')
            matfile = sio.loadmat(foldername + '/microstructureInformation' + str(mesh_name_iter) + '.mat')
            diskCenters = matfile['diskCenters']
            diskRadii = matfile['diskRadii']
            diskRadii = diskRadii.flatten()
            print('Non-overlapping disks loaded.')
            domain = pm.substractCircles(diskCenters, diskRadii)

            # Generate mesh - this step is expensive
            print('Disks substracted.')
            try:
                mesh = pm.generateMesh(domain, nElements)
                print('mesh generated.')
                # save mesh in xml format for later use
                filename = foldername + '/mesh' + str(mesh_name_iter) + '.xml'
                mesh_file = df.File(filename)
                mesh_file << mesh

                # save vertex coordinates and cell connectivity to mat file for easy read-in to matlab
                sio.savemat(foldername + '/mesh' + str(mesh_name_iter) + '.mat',
                            {'x': mesh.coordinates(), 'cells': mesh.cells() + 1.0})
                os.remove(foldername + '/microstructureInformation_nomesh' + str(mesh_name_iter) + '.mat')
            except:
                print('Mesh generation failed. Trying again...')
                # the current microstructure seems to be invalid for mesh generation. Exchange the current
                # microstructure by the very last one generated
                last_microstruct = 0
                while os.path.isfile(foldername + '/microstructureInformation' + str(last_microstruct) + '.mat') or\
                    os.path.isfile(foldername + '/microstructureInformation_nomesh' + str(last_microstruct) + '.mat'):
                    last_microstruct += 1
                os.rename(foldername + '/microstructureInformation_nomesh' + str(last_microstruct) + '.mat',
                         foldername + '/microstructureInformation_nomesh' + str(mesh_name_iter) + '.mat')
                os.remove(foldername + '/microstructureInformation' + str(mesh_name_iter) + '.mat')

    else:
        mesh_name_iter = 0
        while mesh_name_iter < nMeshes:
            if nExclusionsDist == 'uniform':
                nExclusions = np.random.randint(nExclusionParams[0], nExclusionParams[1])
            elif nExclusionsDist == 'logn':
                nExclusions = np.random.lognormal(nExclusionParams[0], nExclusionParams[1])
            nExclusions_upper_bound = 30000         # this is needed to avoid segfaults in mesh generation
            nExclusions = min(nExclusions, nExclusions_upper_bound)
            print('nExclusions = ', nExclusions)
            diskCenters = np.empty([0, 2])
            diskRadii = np.empty([0, 1])
            if substractCorners:
                diskCenters = np.append(diskCenters, np.array([[0.0, 0.0]]), axis=0)
                diskCenters = np.append(diskCenters, np.array([[1.0, 0.0]]), axis=0)
                diskCenters = np.append(diskCenters, np.array([[0.0, 1.0]]), axis=0)
                diskCenters = np.append(diskCenters, np.array([[1.0, 1.0]]), axis=0)
                diskRadii = np.append(diskRadii, 0.03)
                diskRadii = np.append(diskRadii, 0.03)
                diskRadii = np.append(diskRadii, 0.03)
                diskRadii = np.append(diskRadii, 0.03)

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

                overlap = np.any((exclusionRadius + diskRadii) >=
                                 np.linalg.norm(exclusionCenter - diskCenters, axis=1))

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
                            diskCenters = np.append(diskCenters, exclusionCenter, axis=0)
                            diskRadii = np.append(diskRadii, exclusionRadius)
                            currentExclusions += 1
                t_elapsed = time.time() - t_start
            print('Non-overlapping disks drawn.')
            print('real number of exclusions == ', diskRadii.size)
            print('Smallest radius == ', np.amin(diskRadii))

            domain = pm.substractCircles(diskCenters, diskRadii)

            # Generate mesh - this step is expensive
            print('Disks substracted.')
            try:
                mesh = pm.generateMesh(domain, nElements)
                print('mesh generated.')
                # save mesh in xml format for later use
                # check how many meshes already exist and name the mesh accordingly
                # (allows parallel mesh generation jobs)
                mesh_name_iter = 0
                while os.path.isfile(foldername + '/mesh' + str(mesh_name_iter) + '.xml'):
                    mesh_name_iter += 1
                # save microstructural information, i.e. centers and radii of disks
                sio.savemat(foldername + '/microstructureInformation' + str(mesh_name_iter) + '.mat',
                            {'diskCenters': diskCenters, 'diskRadii': diskRadii})

                filename = foldername + '/mesh' + str(mesh_name_iter) + '.xml'
                mesh_file = df.File(filename)
                mesh_file << mesh

                # save vertex coordinates and cell connectivity to mat file for easy read-in to matlab
                sio.savemat(foldername + '/mesh' + str(mesh_name_iter) + '.mat',
                            {'x': mesh.coordinates(), 'cells': mesh.cells() + 1.0})
                os.remove(foldername + '/microstructureInformation_nomesh' + str(mesh_name_iter) + '.mat')
            except:
                print('Mesh generation failed. Trying again...')


