"""This demo solves the Stokes equations using an iterative linear solver.
Note that the sign for the pressure has been flipped for symmetry."""


import matplotlib.pyplot as plt
import numpy as np
import dolfin as df
from randomFieldGeneration import RandomField as rf
import mshr
import time
import scipy.stats as stats
from skimage import measure


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
volumeFraction = .7
cutoff = stats.norm.ppf(volumeFraction)
print('cutoff = ', cutoff)
nMeshPolygon = 128   # image discretization of random material; needed for polygones



randomFieldObj = rf()
randomFieldObj.covarianceFunction = 'sincSq'
randomField = randomFieldObj.sample()

print('Discretizing random field...')
x = np.linspace(0, 1, nMeshPolygon)
img = np.zeros([nMeshPolygon, nMeshPolygon])
for i in range(0, nMeshPolygon):
    for j in range(0, nMeshPolygon):
        img[i, j] = randomField(np.array([x[i], x[j]]))
print('done.')

print('Drawing polygones...')
contours = measure.find_contours(img, cutoff, positive_orientation='high', fully_connected='high')
print('done.')

#  Show image
showImg = False
if showImg:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(img >= cutoff, cmap=plt.cm.inferno)
    for i in range(0, len(contours)):
        contour = contours[i]
        plt.plot(contour[:, 1], contour[:, 0])


domain = mshr.Rectangle(df.Point(0.0, 0.0), df.Point(1.0, 1.0))
print('Substracting blobs as polygones from domain...')
for blob in range(0, len(contours)):
    contour = contours[blob]
    contour = contour/(nMeshPolygon - 1)
    vertexList = []
    for i in range(0, contour.shape[0]):
        # Construct list of df.Point's for polygon vertices
        x = np.array([contour[i, 0], contour[i, 1]])
        vertexList.append(df.Point(np.squeeze(x)))
    # Substract polygon from domain
    try:
        mPolygon = mshr.Polygon(vertexList)
    except RuntimeError:
        print('Invalid blob at')
        print('blob = ', blob)
        print('contour = ', contour)
    domain -= mPolygon
print('done.')

print('generating FE mesh...')
t = time.time()
mesh = mshr.generate_mesh(domain, 128)
elapsed_time = time.time() - t
print('done. Time: ', elapsed_time)



print('Setting boundary conditions...')
# Create classes for defining parts of the boundaries and the interior
# of the domain
class DomainBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS \
                       or x[0] > (1.0 - df.DOLFIN_EPS) or x[0] < df.DOLFIN_EPS

class UpDown(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS

class LeftRight(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > (1.0 - df.DOLFIN_EPS) or x[0] < df.DOLFIN_EPS

class RandField(df.SubDomain):
    def inside(self, x, on_boundary):
        outerBoundary = x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS \
            or x[0] > (1.0 - df.DOLFIN_EPS) or x[0] < df.DOLFIN_EPS
        return on_boundary and not outerBoundary


# Initialize sub-domain instances
domainBoundary = DomainBoundary()
upDown = UpDown()
leftRight = LeftRight()
solidPhase = RandField()

# Mark subdomains
domains = df.CellFunction("size_t", mesh)
domains.set_all(0)
solidPhase.mark(domains, 1)

# Mark boundaries
boundaries = df.FacetFunction("size_t", mesh)
boundaries.set_all(0)
# domainBoundary.mark(boundaries, 1)
upDown.mark(boundaries, 1)
leftRight.mark(boundaries, 2)

# Define mixed function space (Taylor-Hood)
u_e = df.VectorElement("CG", mesh.ufl_cell(), 2)
p_e = df.FiniteElement("CG", mesh.ufl_cell(), 1)
mixedEl = df.MixedElement([u_e, p_e])
W = df.FunctionSpace(mesh, mixedEl)

# Flow boundary condition for velocity on domain boundary
flowFieldLR = df.Expression(('0.0', '0.0'), degree=2, mu=mu)
flowFieldUD = df.Expression(("0.0", "-1.0"), degree=2, mu=mu)
# pressureField = df.Expression("0.0", degree=2, mu=mu)
bc1 = df.DirichletBC(W.sub(0), flowFieldLR, leftRight)
bc2 = df.DirichletBC(W.sub(0), flowFieldUD, upDown)
# bc3 = df.DirichletBC(W.sub(1), pressureField, domainBoundary)

# No-slip boundary condition for velocity on material interfaces
noslip = df.Constant((0.0, 0.0))

# Boundary conditions for solid phase
bc4 = df.DirichletBC(W.sub(0), noslip, solidPhase)

# Collect boundary conditions
bcs = [bc1, bc2, bc4]
print('done.')


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
print('Solving equation system...')
t = time.time()
U = df.Function(W)
solver.solve(U.vector(), bb)
elapsed_time = time.time() - t
print('done. Time: ', elapsed_time)

# Get sub-functions
u, p = U.split()


# Save solution as text files
pMesh = p.compute_vertex_values(mesh)
uMesh = u.compute_vertex_values(mesh)
print('uMesh vertices shape: ', uMesh.shape)
x = mesh.coordinates()

np.savetxt('./data/pressureField', pMesh, delimiter=' ')
np.savetxt('./data/velocityField', uMesh, delimiter=' ')
np.savetxt('./data/coordinates', x, delimiter=' ')

# Save solution in VTK format
ufile_pvd = df.File("./data/velocity.pvd")
ufile_pvd << u
pfile_pvd = df.File("./data/pressure.pvd")
pfile_pvd << p



mx = np.max(pMesh)
mn = np.min(pMesh)
fig = plt.figure()
pp = df.plot(p)
plt.colorbar(pp)
fig = plt.figure()
df.plot(u, cmap=plt.cm.viridis, headwidth=0.005, headlength=0.005, scale=80.0, minlength=0.0001,
        width=0.0008, minshaft=0.01, headaxislength=0.1)
for i in range(0, len(contours)):
    contour = contours[i]/(nMeshPolygon - 1.0)
    plt.plot(contour[:, 0], contour[:, 1], 'k')
plt.xticks([])
plt.yticks([])
fig.savefig('velocity.pdf')

# df.plot(mesh)

plt.show()
