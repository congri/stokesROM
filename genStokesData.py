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
import porousMedia as pm


# Test for PETSc or Epetra
if not df.has_linear_algebra_backend("PETSc") and not df.has_linear_algebra_backend("Epetra"):
    df.info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

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
nElements = 128
nMeshPolygon = 64
covarianceFunction = 'matern'
randFieldParams = [5.0]
lengthScale = [.008, .008]
meshes = np.arange(0, 1)  # vector of random meshes to load

folderbase = '/home/constantin/cluster'
foldername = folderbase + '/python/data/stokesEquation/meshes/meshSize=' + str(nElements) +\
    '/randFieldDiscretization=' + str(nMeshPolygon) + '/cov=' + covarianceFunction +\
    '/params=' + str(randFieldParams) + '/l=' + str(lengthScale[0]) + '_' +\
    str(lengthScale[1]) + '/volfrac=' + str(volumeFraction)


# Set external boundaries of domain
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

# Initialize sub-domain instances for outer domain boundaries
domainBoundary = DomainBoundary()
upDown = UpDown()
leftRight = LeftRight()


for meshNumber in meshes:
    # load mesh from file
    mesh = df.Mesh(foldername + '/mesh' + str(meshNumber) + '.xml')

    print('Setting boundary conditions...')
    # Define interior boundaries
    class RandField(df.SubDomain):
        def inside(self, x, on_boundary):
            outerBoundary = x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS \
                or x[0] > (1.0 - df.DOLFIN_EPS) or x[0] < df.DOLFIN_EPS
            return on_boundary and not outerBoundary

    # Initialize sub-domain instance for interior boundaries
    solidPhase = RandField()

    # Mark boundaries
    boundaries = df.FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    # domainBoundary.mark(boundaries, 1)
    upDown.mark(boundaries, 1)
    leftRight.mark(boundaries, 2)


    vertices = df.VertexFunction('size_t', mesh)
    vertices.set_all(0)
    solidPhase.mark(vertices, 1)


    # Define mixed function space (Taylor-Hood)
    u_e = df.VectorElement("CG", mesh.ufl_cell(), 2)
    p_e = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    mixedEl = df.MixedElement([u_e, p_e])
    W = df.FunctionSpace(mesh, mixedEl)


    # Flow boundary condition for velocity on domain boundary
    flowFieldLR = df.Expression(('1.0', '0.0'), degree=2)
    flowFieldUD = df.Expression(('0.0', '0.0'), degree=2)

    bc1 = df.DirichletBC(W.sub(0), flowFieldLR, leftRight)
    bc2 = df.DirichletBC(W.sub(0), flowFieldUD, upDown)

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
    f = df.Constant((0.0, 0.0)) # right hand side
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

    df.plot(mesh)

    mx = np.max(pMesh)
    mn = np.min(pMesh)
    fig = plt.figure()
    pp = df.plot(p)
    plt.colorbar(pp)
    fig = plt.figure()
    df.plot(u, cmap=plt.cm.viridis, headwidth=0.005, headlength=0.005, scale=80.0, minlength=0.0001,
            width=0.0008, minshaft=0.01, headaxislength=0.1)
    # plot internal boundaries (boundary vertices
    bmesh = df.BoundaryMesh(mesh, 'exterior')
    xBoundary = bmesh.coordinates()
    plt.plot(xBoundary[:, 0], xBoundary[:, 1], 'ko', ms=.5)
    

    plt.xticks([])
    plt.yticks([])
    fig.savefig('velocity.pdf')

plt.show()
