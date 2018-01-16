"""This demo solves the Stokes equations, using quadratic elements for
the velocity and first degree elements for the pressure (Taylor-Hood
elements). The sub domains for the different boundary conditions
used in this simulation are computed by the demo program in
src/demo/mesh/subdomains."""

# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008-2009.
#
# First added:  2007-11-16
# Last changed: 2009-11-26
# Begin demo

from __future__ import print_function
import dolfin as df
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np


# Load mesh and subdomains
mesh = df.Mesh("./dolfin_fine.xml.gz")
sub_domains = df.MeshFunction("size_t", mesh, "./dolfin_fine_subdomains.xml.gz")

df.plot(mesh)
df.plot(sub_domains)

class UpDown(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS

upDown = UpDown()
'''
# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q
'''

# Define mixed function space (Taylor-Hood)
u_e = df.VectorElement("CG", mesh.ufl_cell(), 2)
p_e = df.FiniteElement("CG", mesh.ufl_cell(), 1)
mixedEl = df.MixedElement([u_e, p_e])
W = df.FunctionSpace(mesh, mixedEl)

# No-slip boundary condition for velocity 
# x1 = 0, x1 = 1 and around the dolphin
noslip = df.Constant((0, 0))
bc0 = df.DirichletBC(W.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
# x0 = 1
inflow = df.Expression(("-sin(x[1]*pi)", "0.0"), degree=2)
bc1 = df.DirichletBC(W.sub(0), inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
# x0 = 0
zero = df.Constant(0)
bc2 = df.DirichletBC(W.sub(1), zero, sub_domains, 2)

# Collect boundary conditions
bcs = [bc0, bc1]

# Define variational problem
(u, p) = df.TrialFunctions(W)
(v, q) = df.TestFunctions(W)
f = df.Constant((0, 0))
a = (df.inner(df.grad(u), df.grad(v)) - df.div(v)*p + q*df.div(u))*df.dx
L = df.inner(f, v)*df.dx

# Compute solution
w = df.Function(W)
df.solve(a == L, w, bcs)

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)

print("Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2"))
print("Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2"))

# # Split the mixed solution using a shallow copy
(u, p) = w.split()

'''
# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p


# Save solution to mat file for easy read-in in matlab
sio.savemat('./solutionDolfin.mat',
    {'u': np.reshape(u.compute_vertex_values(), (2, -1)), 'p': p.compute_vertex_values(),
    'x': mesh.coordinates(), 'cells': mesh.cells() + 1.0})

# Plot solution
plot(u)
plot(p)
interactive()
plt.show()
'''
