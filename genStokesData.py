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
import subprocess as sp


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
volumeFraction = .9
nElements = 128
nMeshPolygon = 64
covarianceFunction = 'matern'
randFieldParams = [5.0]
lengthScale = [.008, .008]
meshes = np.arange(0, 128)  # vector of random meshes to load

folderbase = '/home/constantin/cluster'
foldername = folderbase + '/python/data/stokesEquation/meshes/meshSize=' + str(nElements) +\
    '/randFieldDiscretization=' + str(nMeshPolygon) + '/cov=' + covarianceFunction +\
    '/params=' + str(randFieldParams) + '/l=' + str(lengthScale[0]) + '_' +\
    str(lengthScale[1]) + '/volfrac=' + str(volumeFraction)


# Set external boundaries of domain
class DomainBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS\
               or x[0] > (1.0 - df.DOLFIN_EPS) or x[0] < df.DOLFIN_EPS


class UpDown(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS


class LeftRight(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > (1.0 - df.DOLFIN_EPS) or x[0] < df.DOLFIN_EPS


class Origin(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < df.DOLFIN_EPS and x[1] < df.DOLFIN_EPS


# Initialize sub-domain instances for outer domain boundaries
domainBoundary = DomainBoundary()
upDown = UpDown()
leftRight = LeftRight()
origin = Origin()


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

    # Define mixed function space (Taylor-Hood)
    u_e = df.VectorElement("CG", mesh.ufl_cell(), 2)
    p_e = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    mixedEl = df.MixedElement([u_e, p_e])
    W = df.FunctionSpace(mesh, mixedEl)

    # Flow boundary condition for velocity on domain boundary
    flowFieldLR = df.Expression(('(x[1] - .5)*(x[1] - .5) - .25', '0.0'), degree=2)
    flowFieldUD = df.Expression(('0.0', '0.0'), degree=2)

    bc1 = df.DirichletBC(W.sub(0), flowFieldLR, leftRight)
    bc2 = df.DirichletBC(W.sub(0), flowFieldUD, upDown)

    # No-slip boundary condition for velocity on material interfaces
    noslip = df.Constant((0.0, 0.0))

    # Boundary conditions for solid phase
    bc3 = df.DirichletBC(W.sub(0), noslip, solidPhase)

    #pressure boundary condition
    zero_p = df.Constant(0.0)
    bc4 = df.DirichletBC(W.sub(1), zero_p, origin, method='pointwise')

    # Collect boundary conditions
    bcs = [bc1, bc2, bc3, bc4]
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
    try:
        solver.solve(U.vector(), bb)
        elapsed_time = time.time() - t
        print('done. Time: ', elapsed_time)

        # Get sub-functions
        u, p = U.split()

        # Save solution in VTK format, same folder as mesh
        saveVelocityFile = foldername + '/velocity' + str(meshNumber) + '.pvd'
        savePressureFile = foldername + '/pressure' + str(meshNumber) + '.pvd'
        ufile_pvd = df.File(saveVelocityFile)
        ufile_pvd << u
        pfile_pvd = df.File(savePressureFile)
        pfile_pvd << p

        # save full function space U to xml
        saveFullSolutionFile = foldername + '/fullSolution' + str(meshNumber) + '.xml'
        Ufile = df.File(saveFullSolutionFile)
        Ufile << U

        plot_flag = True
        if plot_flag:
            '''
            df.plot(mesh)

            fig = plt.figure()
            pp = df.plot(p)
            plt.colorbar(pp)
            '''
            fig = plt.figure()
            df.plot(u, cmap=plt.cm.viridis, headwidth=0.005, headlength=0.005, scale=80.0, minlength=0.0001,
                    width=0.0008, minshaft=0.01, headaxislength=0.1)
            # plot internal boundaries (boundary vertices
            bmesh = df.BoundaryMesh(mesh, 'exterior')
            xBoundary = bmesh.coordinates()
            plt.plot(xBoundary[:, 0], xBoundary[:, 1], 'ko', ms=.5)


            plt.xticks([])
            plt.yticks([])
            velocityFigureFile = foldername + '/velocity' + str(meshNumber) + '.pdf'
            fig.savefig(velocityFigureFile)
            sp.run(['pdfcrop', velocityFigureFile, velocityFigureFile])
            plt.close(fig)
    except:
        print('Solver failed to converge. Passing to next mesh')


'''
if plot_flag:
    plt.show()
'''
