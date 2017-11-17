"""This demo solves the Stokes equations using an iterative linear solver.
Note that the sign for the pressure has been flipped for symmetry."""


import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import dolfin as df
from randomFieldGeneration import RandomField as rf

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
mu = 1  # viscosity
cutoff = 1

randomField = rf.sample(rf)

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
def solidPhase(x, on_boundary): return randomField(x) > cutoff


# No-slip boundary condition for velocity
noslip = df.Constant((0.0, 0.0))
bc0 = df.DirichletBC(W.sub(0), noslip, top_bottom)

# Inflow boundary condition for velocity
inflow = df.Expression(("2*sin(2*pi*x[1])", "0.0"), degree=2, mu=mu)
bc1 = df.DirichletBC(W.sub(0), inflow, right)

# Boundary condition at outflow
outflow = df.Expression(("sin(2*pi*x[1])", "0.0"), degree=2, mu=mu)
bc2 = df.DirichletBC(W.sub(0), outflow, left)

# Boundary conditions for solid phase
zero_p = df.Constant(0)
noflow = df.Expression(("0.0", "0.0"), degree=2)
bc3 = df.DirichletBC(W.sub(0), noflow, solidPhase)
# Set pressure to 0 in no flow regions
bc4 = df.DirichletBC(W.sub(1), zero_p, solidPhase)

# Collect boundary conditions
bcs = [bc0, bc1, bc2, bc3, bc4]

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

# Save solution as text files
pMesh = p.compute_vertex_values(mesh)
uMesh = u.compute_vertex_values(mesh)
print(uMesh.shape)
x = mesh.coordinates()

cond = np.zeros(x.shape[0])
for i in range(0, x.shape[0]):
    cond[i] = randomField(np.transpose(x[i, :]))

cond = np.reshape(cond, [nMesh + 1, nMesh + 1])
condBin = cond > cutoff

# Plot cond with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(cond, extent=[0, 1, 0, 1])
cbar = fig.colorbar(cax)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(condBin, extent=[0, 1, 0, 1])
cbar = fig.colorbar(cax)

np.savetxt('./data/pressureField', pMesh, delimiter=' ')
np.savetxt('./data/velocityField', uMesh, delimiter=' ')
np.savetxt('./data/coordinates', x, delimiter=' ')


# Save solution in VTK format
ufile_pvd = df.File("./data/velocity.pvd")
ufile_pvd << u
pfile_pvd = df.File("./data/pressure.pvd")
pfile_pvd << p


# Plot with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x[0:(nMesh + 1), 0], x[0:(nMesh + 1), 0])
pMesh = np.reshape(pMesh, (nMesh + 1, nMesh + 1))
pMesh[condBin] = np.ma.masked

ax.plot_surface(X, Y, pMesh, cmap=cm.inferno)

plt.show()
