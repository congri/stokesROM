"""This demo solves the Stokes equations using an iterative linear solver.
Note that the sign for the pressure has been flipped for symmetry."""


import matplotlib.pyplot as plt
import numpy as np
import dolfin as df
import time
import subprocess as sp
import scipy.io as sio
import os


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

# general parameters
meshes = np.arange(0, 128)  # vector of random meshes to load
porousMedium = 'nonOverlappingCircles'    #circles or randomField
nElements = 128

# Define physical parameters
mu = 1  # viscosity

# For random fields
volumeFraction = .8
nMeshPolygon = 64
covarianceFunction = 'se'
randFieldParams = [5.0]
lengthScale = [.008, .008]

# For circular exclusions
nExclusionsMin = 128
nExclusionsMax = 1025
coordinateDistribution = 'uniform'
radiiDistribution = 'uniform'
# to avoid circles on boundaries. Min. distance of circle centers to (lo., r., u., le.) boundary
margins = (0, .03, 0, .03)
r_params = (.003, .015)


# Flow boundary condition for velocity on domain boundary
u_x = '0.25 - (x[1] - 0.5)*(x[1] - 0.5)'
u_y = '0.0'
flowField = df.Expression((u_x, u_y), degree=2)


folderbase = '/home/constantin/cluster'
foldername = folderbase + '/python/data/stokesEquation/meshes/meshSize=' + str(nElements)
if porousMedium == 'randomField':
    foldername = foldername + '/randFieldDiscretization=' + str(nMeshPolygon) + '/cov=' + covarianceFunction +\
    '/params=' + str(randFieldParams) + '/l=' + str(lengthScale[0]) + '_' +\
    str(lengthScale[1]) + '/volfrac=' + str(volumeFraction)
elif porousMedium == 'circles':
    foldername = foldername + '/nCircExcl=' + str(nExclusionsMin) + '-' + str(nExclusionsMax) +\
                 '/coordDist=' + coordinateDistribution + '_margins=' + str(margins) + '/radiiDist=' +\
                 radiiDistribution + '_r_params=' + str(r_params)
elif porousMedium == 'nonOverlappingCircles':
    foldername = foldername + '/nNonOverlapCircExcl=' + str(nExclusionsMin) + '-' + str(nExclusionsMax) +\
        '/coordDist=' + coordinateDistribution + '_margins=' + str(margins) + '/radiiDist=' +\
        radiiDistribution + '_r_params=' + str(r_params)


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
    print('Loading mesh...')
    mesh = df.Mesh(foldername + '/mesh' + str(meshNumber) + '.xml')
    print('mesh loaded.')

    print('Setting boundary conditions...')

    # Define interior boundaries
    class InteriorBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            outerBoundary = x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS \
                or x[0] > (1.0 - df.DOLFIN_EPS) or x[0] < df.DOLFIN_EPS
            return on_boundary and not outerBoundary

    # Initialize sub-domain instance for interior boundaries
    interiorBoundary = InteriorBoundary()

    # Define mixed function space (Taylor-Hood)
    u_e = df.VectorElement("CG", mesh.ufl_cell(), 2)
    p_e = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    mixedEl = df.MixedElement([u_e, p_e])
    W = df.FunctionSpace(mesh, mixedEl)


    # No-slip boundary condition for velocity on material interfaces
    noslip = df.Constant((0.0, 0.0))
    # Boundary conditions for solid phase
    bc1 = df.DirichletBC(W.sub(0), noslip, interiorBoundary)

    # BC's on domain boundary
    bc2 = df.DirichletBC(W.sub(0), flowField, domainBoundary)


    # Collect boundary conditions
    bcs = [bc1, bc2]
    print('boundary conditions set.')


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
        print('equation system solved. Time: ', elapsed_time)
        print('sample: ', meshNumber)

        # Get sub-functions
        u, p = U.split()

        '''
        # Save solution in VTK format, same folder as mesh
        saveVelocityFile = foldername + '/velocity' + str(meshNumber) + '.pvd'
        savePressureFile = foldername + '/pressure' + str(meshNumber) + '.pvd'
        ufile_pvd = df.File(saveVelocityFile)
        ufile_pvd << u
        pfile_pvd = df.File(savePressureFile)
        pfile_pvd << p
        '''

        # save full function space U to xml
        '''
        Ufile = df.File(foldername + '/u_x=' + u_x + '_u_y=' + u_y + '/solution' + str(meshNumber) + '.xml')
        Ufile << U
        '''

        # Save solution to mat file for easy read-in in matlab
        solutionfolder = foldername + '/u_x=' + u_x + '_u_y=' + u_y
        if not os.path.exists(solutionfolder):
            os.makedirs(solutionfolder)

        sio.savemat(solutionfolder + '/solution' + str(meshNumber) + '.mat',
                    {'u': np.reshape(u.compute_vertex_values(), (2, -1)), 'p': p.compute_vertex_values(),
                     'x': mesh.coordinates()})

        plot_flag = False
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


