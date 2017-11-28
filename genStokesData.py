"""This demo solves the Stokes equations using an iterative linear solver.
Note that the sign for the pressure has been flipped for symmetry."""


import matplotlib.pyplot as plt
import numpy as np
import dolfin as df
from randomFieldGeneration import RandomField as rf
import mshr
import time
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
cutoff = 1.0

randomFieldObj = rf()
randomField = randomFieldObj.sample()
'''
print('Drawing polygones...')
nMesh = 64
x = np.linspace(0, 1, nMesh)
img = np.zeros([nMesh, nMesh])
for i in range(0, nMesh):
    for j in range(0, nMesh):
        img[i, j] = randomField(np.array([x[i], x[j]]))

ic = iso.IsoContour()
Objects, Vertices, _ = ic.isocontour(img, cutoff)
print('done.')
'''

print('Drawing polygones...')
nMesh = 128
x = np.linspace(0, 1, nMesh)
img = np.zeros([nMesh, nMesh])
for i in range(0, nMesh):
    for j in range(0, nMesh):
        img[i, j] = randomField(np.array([x[i], x[j]]))
contours = measure.find_contours(img, cutoff)
print('done.')

#  Show image
showImg = True
if showImg:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(img >= cutoff, cmap=plt.cm.inferno)
    for i in range(0, len(contours)):
        contour = contours[i]
        plt.plot(contour[:, 1], contour[:, 0])



domain = mshr.Rectangle(df.Point(0.0, 0.0), df.Point(1.0, 1.0))

'''
print('Substracting polygones from domain...')
for blob in range(0, len(Objects)):
    polygonVertices = np.flip(Objects[blob], axis=0)
    # Construct polygon
    vertexList = []
    print('Vertices = ', polygonVertices)
    print('Coordinates = ', Vertices[polygonVertices, :])
    for i in range(0, polygonVertices.size):
        # Construct list of df.Point's for polygon vertices
        vertex = polygonVertices[i]
        x = np.array([Vertices[vertex, 0], Vertices[vertex, 1]])
        vertexList.append(df.Point(np.squeeze(x)))
    # Substract polygon from domain
    try:
        domain -= mshr.Polygon(vertexList)
    except RuntimeError:
        print('blob = ', blob)
        print('vertices = ', polygonVertices)
        print('coordinates = ', Vertices[polygonVertices])
print('done.')
'''

print('Substracting polygones from domain...')
for blob in range(0, len(contours)):
    contour = contours[blob]
    contour = np.flip(contour, axis=0)/(nMesh - 1)
    vertexList = []
    for i in range(0, contour.shape[0]):
        # Construct list of df.Point's for polygon vertices
        x = np.array([contour[i, 0], contour[i, 1]])
        vertexList.append(df.Point(np.squeeze(x)))
    # Substract polygon from domain
    try:
        domain -= mshr.Polygon(vertexList)
    except RuntimeError:
        print('blob = ', blob)
        print('contour = ', contour)
print('done.')


print('generating FE mesh...')
t = time.time()
mesh = mshr.generate_mesh(domain, 128)
#mesh = df.UnitSquareMesh(128, 128)
elapsed_time = time.time() - t
print('done. Time: ', elapsed_time)





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


# Initialize mesh function for interior domains
domains = df.CellFunction("size_t", mesh)
domains.set_all(0)
solidPhase.mark(domains, 1)

# Initialize mesh function for boundary domains
boundaries = df.FacetFunction("size_t", mesh)
boundaries.set_all(0)
# domainBoundary.mark(boundaries, 1)
upDown.mark(boundaries, 1)
leftRight.mark(boundaries, 2)


# Define mixed function space
u_e = df.VectorElement("CG", mesh.ufl_cell(), 2)
p_e = df.FiniteElement("CG", mesh.ufl_cell(), 1)
mixedEl = df.MixedElement([u_e, p_e])
W = df.FunctionSpace(mesh, mixedEl)


# Flow boundary condition for velocity on domain boundary
flowFieldLR = df.Expression(('1', '0'), degree=2, mu=mu)
flowFieldUD = df.Expression(("0.0", "0.0"), degree=2, mu=mu)
#pressureField = df.Expression("0.0", degree=2, mu=mu)
bc1 = df.DirichletBC(W.sub(0), flowFieldLR, leftRight)
bc2 = df.DirichletBC(W.sub(0), flowFieldUD, upDown)
#bc3 = df.DirichletBC(W.sub(1), pressureField, domainBoundary)

# No-slip boundary condition for velocity on material interfaces
noslip = df.Constant((0.0, 0.0))
zero_p = df.Constant(0.0)

# Boundary conditions for solid phase
bc4 = df.DirichletBC(W.sub(0), noslip, solidPhase)
bc5 = df.DirichletBC(W.sub(1), zero_p, solidPhase)

# Collect boundary conditions
bcs = [bc1, bc2, bc4]


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


'''
cond = np.zeros([100, 100])
pMesh = np.zeros([100, 100])
X = np.zeros([100, 100])
Y = np.zeros([100, 100])
for i in range(0, 100):
    for j in range(0, 100):
        x = [i/100, j/100]
        cond[i, j] = randomField(x)
        pMesh[i, j] = p(x)
        X[i, j] = x[0]
        Y[i, j] = x[1]

# cond = np.reshape(cond, [nMesh + 1, nMesh + 1])
condBin = cond > cutoff

# Plot cond with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.contourf(X, Y, cond, extent=[0, 1, 0, 1], cmap=cm.inferno)
cbar = fig.colorbar(cax)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cax = ax.plot_surface(X, Y, pMesh, cmap=cm.inferno)
cbar = fig.colorbar(cax)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.contourf(X, Y, condBin, extent=[0, 1, 0, 1], cmap=cm.binary)
cbar = fig.colorbar(cax)
'''





np.savetxt('./data/pressureField', pMesh, delimiter=' ')
np.savetxt('./data/velocityField', uMesh, delimiter=' ')
np.savetxt('./data/coordinates', x, delimiter=' ')

# Save solution in VTK format
ufile_pvd = df.File("./data/velocity.pvd")
ufile_pvd << u
pfile_pvd = df.File("./data/pressure.pvd")
pfile_pvd << p


'''
# Plot with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x[0:(nMesh + 1), 0], x[0:(nMesh + 1), 0])
pMesh = np.reshape(pMesh, (nMesh + 1, nMesh + 1))
# pMesh[condBin] = np.ma.masked
ax.plot_surface(X, Y, pMesh, cmap=cm.inferno)

# Plot cond with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(pMesh, extent=[0, 1, 0, 1], cmap=cm.inferno)
cbar = fig.colorbar(cax)
'''




mx = np.max(pMesh)
mn = np.min(pMesh)
fig = plt.figure()
pp = df.plot(p)
plt.colorbar(pp)
fig = plt.figure()
df.plot(u, cmap=plt.cm.inferno)
fig = plt.figure()
df.plot(mesh)


plt.show()
