'''This is a script to generate and save meshes'''

import scipy.stats as stats
from randomFieldGeneration import RandomField as rf
import porousMedia as pm
import dolfin as df
import os



def potential(x):
    # Potential to avoid blobs on boundary
    p = 0
    # Penalty for boundaries at 0
    sigma = 1e-3
    prefactor = 1.0
    p -= prefactor*stats.laplace.pdf(x[0], 0, sigma)
    p -= prefactor*stats.laplace.pdf(x[1], 0, sigma)
    p -= prefactor*stats.laplace.pdf(x[0], 1, sigma)
    p -= prefactor*stats.laplace.pdf(x[1], 1, sigma)
    return p

def addfunctions(f, g):
    # add functions f and g
    def h(x):
        return f(x) + g(x)
    return h


volumeFraction = .7 # Volume fraction of voids
cutoff = stats.norm.ppf(volumeFraction)
print('cutoff = ', cutoff)
nMeshPolygon = 64   # image discretization of random material; needed for polygones
nMeshes = 16
nElements = 128 # PDE discretization


randomFieldObj = rf()
randomFieldObj.covarianceFunction = 'matern'

foldername = '/home/constantin/python/data/stokesEquation/meshes/meshSize=' + str(nElements) +\
    '/randFieldDiscretization=' + str(nMeshPolygon) + '/cov=' + randomFieldObj.covarianceFunction +\
    '/params=' + str(randomFieldObj.params) + '/l=' + str(randomFieldObj.lengthScale[0]) + '_' +\
    str(randomFieldObj.lengthScale[1])
if not os.path.exists(foldername):
    os.makedirs(foldername)

for i in range(0, nMeshes):
    randomField = randomFieldObj.sample()
    randomField = addfunctions(randomField, potential)
    discretizedRandomField = pm.discretizeRandField(randomField,
                                                nDiscretize=(nMeshPolygon, nMeshPolygon))

    contours = pm.findPolygones(discretizedRandomField, cutoff)
    contours = pm.rescalePolygones(contours, nDiscretize=(nMeshPolygon, nMeshPolygon))
    domain = pm.substractPores(contours)

    # Generate mesh - this step is expensive
    mesh = pm.generateMesh(domain)

    filename = foldername + '/mesh' + str(i) + '.xml'
    mesh_file = df.File(filename)
    mesh_file << mesh
