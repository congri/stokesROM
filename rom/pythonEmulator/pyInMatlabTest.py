"""Testing to call this from Matlab"""

import dolfin as df
# import dolfin_adjoint as dfa
import numpy as np


# Create mesh and define function space
mesh = df.UnitSquareMesh(4, 4)
V = df.FunctionSpace(mesh, 'CG', 1)
# Vcond = df.FunctionSpace(mesh, 'DG', 0)


def testFun(str):
    print(str)
    print(' aand Hll Wrld')
    return 88.0

