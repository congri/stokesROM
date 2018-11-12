'''This is a script to generate and save meshes on the cluster. It is copied from generateMeshes.py, but only
essential parts are kept to hopefully avoid segfaults'''

import dolfin as df
import os
from shutil import copyfile
import scipy.io as sio
import mshr


# save mesh in xml format for later use
filename = './mesh21'
mesh = df.Mesh(filename + '.xml')

# save vertex coordinates and cell connectivity to mat file for easy read-in to matlab
sio.savemat(filename + '.mat', {'x': mesh.coordinates(), 'cells': mesh.cells() + 1.0})
