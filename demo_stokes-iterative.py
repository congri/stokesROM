"""This demo solves the Stokes equations using an iterative linear solver.
Note that the sign for the pressure has been flipped for symmetry."""

# Copyright (C) 2010 Garth N. Wells
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
# First added:  2010-08-08
# Last changed: 2010-08-08

# Begin demo

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import dolfin as df
# Test for PETSc or Epetra
if not df.has_linear_algebra_backend("PETSc") and not df.has_linear_algebra_backend("Epetra"):
    df.info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

"""
if not df.has_krylov_solver_preconditioner("amg"):
    df.info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
	 "preconditioner, Hypre or ML.")
    exit()
"""

if df.has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif df.has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    df.info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()

# Define physical parameters
mu = 10  #viscosity


# Load mesh
nMesh = 32
mesh = df.UnitSquareMesh(nMesh, nMesh)


# Define mixed function space
u_e = df.VectorElement("CG", mesh.ufl_cell(), 2)
p_e = df.FiniteElement("CG", mesh.ufl_cell(), 1)
mixedEl = df.MixedElement([u_e, p_e])
W = df.FunctionSpace(mesh, mixedEl)


# Boundaries
def right(x, on_boundary): return x[0] > (1.0 - df.DOLFIN_EPS)
def left(x, on_boundary): return x[0] < df.DOLFIN_EPS
def top_bottom(x, on_boundary):
    return x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS


# No-slip boundary condition for velocity
noslip = df.Constant((0.0, 0.0))
bc0 = df.DirichletBC(W.sub(0), noslip, top_bottom)

# Inflow boundary condition for velocity
inflow = df.Expression(("mu*2*sin(2*pi*x[1])", "0.0"), degree=2, mu=mu)
bc1 = df.DirichletBC(W.sub(0), inflow, right)

# Boundary condition for pressure at outflow
outflow_p = df.Constant(0)
outflow = df.Expression(("mu*sin(2*pi*x[1])", "0.0"), degree=2, mu=mu)
bc2 = df.DirichletBC(W.sub(0), outflow, left)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(u, p) = df.TrialFunctions(W)
(v, q) = df.TestFunctions(W)
f = df.Constant((0.0, 0.0))
a = mu*df.inner(df.grad(u), df.grad(v))*df.dx + df.div(v)*p*df.dx + q*df.div(u)*df.dx
L = df.inner(f, v)*df.dx

# Form for use in constructing preconditioner matrix
b = df.inner(df.grad(u), df.grad(v))*df.dx + p*q*df.dx

# Assemble system
A, bb = df.assemble_system(a, L, bcs)

# Assemble preconditioner system
P, btmp = df.assemble_system(b, L, bcs)

# Create Krylov solver and AMG preconditioner
solver = df.KrylovSolver(krylov_method)

# Associate operator (A) and preconditioner matrix (P)
solver.set_operators(A, P)

# Solve
U = df.Function(W)
solver.solve(U.vector(), bb)

# Get sub-functions
u, p = U.split()

pMesh = p.compute_vertex_values(mesh)
uMesh = u.compute_vertex_values(mesh)
x = mesh.coordinates()
np.savetxt('pressureField', pMesh, delimiter=' ')
np.savetxt('velocityField', uMesh, delimiter=' ')
np.savetxt('coordinates', x, delimiter=' ')


# Save solution in VTK format
ufile_pvd = df.File("velocity.pvd")
ufile_pvd << u
pfile_pvd = df.File("pressure.pvd")
pfile_pvd << p

"""
# Plot solution
df.plot(u)
df.plot(p)
df.interactive()
"""


# Plot with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x[0:(nMesh + 1), 0], x[0:(nMesh + 1), 0])
pMesh = np.reshape(pMesh, (nMesh + 1, nMesh + 1))

ax.plot_surface(X, Y, pMesh, cmap=cm.inferno)

plt.show()
