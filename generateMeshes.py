'''This is a script to generate and save meshes'''

import scipy.stats as stats
from randomFieldGeneration import RandomField as rf
import porousMedia as pm
import dolfin as df
import os
import numpy as np

# Global parameters
mode = 'circles'
nMeshes = 8
nElements = 15  # PDE discretization
foldername1 = '/home/constantin/python/data/stokesEquation/meshes/meshSize=' + str(nElements)

if mode == 'randomField':

    print('Generate meshes from discretized random fields...')
    volumeFraction = .7  # Volume fraction of voids
    cutoff = stats.norm.ppf(volumeFraction)
    print('cutoff = ', cutoff)
    nMeshPolygon = 64  # image discretization of random material; needed for polygones


    def potential(x):
        # Potential to avoid blobs on boundary
        p = 0
        # Penalty for boundaries at 0
        sigma = 1e-3
        prefactor = 1.0
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
    randomFieldObj.covarianceFunction = 'matern'

    foldername = foldername1 + '/randFieldDiscretization=' + str(nMeshPolygon) + '/cov=' + \
                 randomFieldObj.covarianceFunction + '/params=' + str(randomFieldObj.params) + '/l=' + \
                 str(randomFieldObj.lengthScale[0]) + '_' + str(randomFieldObj.lengthScale[1]) + \
                 '/volfrac=' + str(volumeFraction)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for i in range(0, nMeshes):
        randomField = randomFieldObj.sample()
        randomField = addfunctions(randomField, potential)
        discretizedRandomField = pm.discretizeRandField(randomField,
                                                        nDiscretize=(nMeshPolygon, nMeshPolygon))

        contours = pm.findPolygones(discretizedRandomField, cutoff)
        contours = pm.rescalePolygones(contours, nDiscretize=(nMeshPolygon, nMeshPolygon))
        domain = pm.substractPolygones(contours)

        # Generate mesh - this step is expensive
        mesh = pm.generateMesh(domain)

        filename = foldername + '/mesh' + str(i) + '.xml'
        mesh_file = df.File(filename)
        mesh_file << mesh


elif mode == 'circles':
    print('Generating mesh with circular exclusions...')

    nExclusions = 512
    coordinateDist = 'uniform'
    c_params = (.022, .978)
    radiiDist = 'uniform'
    r_params = (.005, .02)

    foldername = foldername1 + '/nCircExcl=' + str(nExclusions) + '/coordDist=' +\
                 coordinateDist + '_c_params=' + str(c_params) +\
                 '/radiiDist=' + radiiDist + '_r_params=' + str(r_params)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for i in range(0, nMeshes):
        if coordinateDist == 'uniform':
            exclusionCoordinates = (c_params[1] - c_params[0])*np.random.rand(nExclusions, 2) + c_params[0]

        if radiiDist == 'uniform':
            exclusionRadii = (r_params[1] - r_params[0]) * np.random.rand(nExclusions) + r_params[0]
        elif radiiDist == 'exponential':
            exclusionRadii = np.random.exponential(r_params, nExclusions)

        domain = pm.substractCircles(exclusionCoordinates, exclusionRadii)

        # Generate mesh - this step is expensive
        mesh = pm.generateMesh(domain)

        filename = foldername + '/mesh' + str(i) + '.xml'
        mesh_file = df.File(filename)
        mesh_file << mesh
