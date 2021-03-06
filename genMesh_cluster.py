'''This is a script to generate and save meshes on the cluster. It is copied from generateMeshes.py, but only
essential parts are kept to hopefully avoid segfaults'''

import dolfin as df
import os
from shutil import copyfile
import scipy.io as sio
import mshr


# Global parameters
nMeshes = 2500
nElements = 256  # PDE discretization
foldername1 = '/home/constantin/python/data/stokesEquation/meshSize=' + str(nElements)


# Parameters only for 'circles' mode
nExclusionsDist='logn'
nExclusionParams = [8.1, 0.6]
coordinateDist = 'GP'
# to avoid circles on boundaries. Min. distance of circle centers to (lo., r., u., le.) boundary
# negative margin means no margin
margins = [0.003, 0.003, 0.003, 0.003]
origin_margin = .005
substractCorners = False     # Substracts circles from domain corners s.t. flow cannot pass
radiiDist = 'lognGP'
r_params = [-5.53, 0.3]
# for x~gauss
coordinate_cov = [[0.55, -0.45], [-0.45, 0.55]]
coordinate_mu = [.8, .8]
# for x~GP
covFun = 'squaredExponential'
cov_l = 0.1
sig_scale = 1.5
sigmaGP_r = 0.4
lengthScale_r = .05


# we need to remove '.0' for correct path names
if nExclusionParams[0] % 1 == 0:
    nExclusionParams[0] = int(nExclusionParams[0])
if nExclusionParams[1] % 1 == 0:
    nExclusionParams[1] = int(nExclusionParams[1])
if r_params[0] % 1 == 0:
    r_params[0] = int(r_params[0])
if r_params[1] % 1 == 0:
    r_params[1] = int(r_params[1])
if cov_l % 1 == 0:
    cov_l = int(cov_l)
if sig_scale % 1 == 0:
    sig_scale = int(sig_scale)


print('Generating mesh with non-overlapping circular exclusions...')
foldername = foldername1 + '/nonOverlappingDisks/margins=' + str(margins[0]) + '_' + str(margins[1]) + '_' + \
             str(margins[2]) + '_' + str(margins[3]) + '/N~logn/mu=' + str(nExclusionParams[0]) + '/sigma=' +\
             str(nExclusionParams[1]) + '/x~' + coordinateDist

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

foldername += 'r~' + radiiDist
if radiiDist == 'lognGP':
    foldername += '/mu=' + str(r_params[0]) + '/sigma=' + str(r_params[1]) +\
                  '/sigmaGP_r=' + str(sigmaGP_r) + '/l=' + str(lengthScale_r)
else:
    foldername += '/mu=' + str(r_params[0]) + '/sigma=' + str(r_params[1])
if not os.path.exists(foldername):
    os.makedirs(foldername)


'''first copy 'microstructureInformation_nomesh' to 'microstructureInformation', to give signal that mesh is
generated, so that no other job is taking the same microstructure to generate a mesh'''
mesh_name_iter = 0
while os.path.isfile(foldername + '/microstructureInformation' + str(mesh_name_iter) + '.mat')\
        and mesh_name_iter < nMeshes:
    mesh_name_iter += 1
print('Generating mesh number ', mesh_name_iter)

# copy microstructureInformation file, this is the signal that a job is already generating a mesh
copyfile(foldername + '/microstructureInformation_nomesh' + str(mesh_name_iter) + '.mat',
         foldername + '/microstructureInformation' + str(mesh_name_iter) + '.mat')
os.system('sync')   #to copy asap

print('Loading microstructural data...')
matfile = sio.loadmat(foldername + '/microstructureInformation' + str(mesh_name_iter) + '.mat')
diskCenters = matfile['diskCenters']
diskRadii = matfile['diskRadii']
diskRadii = diskRadii.flatten()
print('... microstructural data loaded.')

# Generate domain object
print('Generating domain object...')
domain = mshr.Rectangle(df.Point(0.0, 0.0), df.Point(1.0, 1.0))
nCircles = diskCenters.shape[0]
for blob in range(0, nCircles):
    c = df.Point(diskCenters[blob, 0], diskCenters[blob, 1])
    domain -= mshr.Circle(c, diskRadii[blob])
print('...domain object generated.')

# Final step: generate mesh using mshr
print('generating FE mesh...')
mesh = mshr.generate_mesh(domain, nElements)
print('...FE mesh generated.')

# save mesh in xml format for later use
# OUTDATED
# filename = foldername + '/mesh' + str(mesh_name_iter) + '.xml'
# print('saving mesh to ./mesh', str(mesh_name_iter), '.xml ...')
# mesh_file = df.File(filename)
# mesh_file << mesh
# print('... ./mesh', str(mesh_name_iter), '.xml saved.')

# save vertex coordinates and cell connectivity to mat file for easy read-in to matlab
print('saving mesh to ./mesh' + str(mesh_name_iter) + '.mat ...')
# is this more efficient with compression turned on?
sio.savemat(foldername + '/mesh' + str(mesh_name_iter) + '.mat',
            {'x': mesh.coordinates(), 'cells': mesh.cells() + 1}, do_compression=True)
print('... ./mesh' + str(mesh_name_iter) + '.mat saved.')
print('removing ' + './microstructureInformation_nomesh' + str(mesh_name_iter) + '.mat ...')
os.remove(foldername + '/microstructureInformation_nomesh' + str(mesh_name_iter) + '.mat')
print('... ./microstructureInformation_nomesh' + str(mesh_name_iter) + '.mat removed.')


