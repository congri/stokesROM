"""This is a cluster-executable version of the Stokes flow PDE solution"""

import numpy as np
import dolfin as df
import time
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
meshes = np.arange(0, 2500)  # vector of random meshes to load
porousMedium = 'nonOverlappingCircles'  # circles or randomField
nElements = 256

# Define physical parameters
mu = 1  # viscosity

# For circular exclusions
nExclusionsDist = 'logn'
nExclusionParams = (8.1, 0.6)
coordinateDistribution = 'GP'

# for coordinateDistribution == 'gauss'
coordinate_cov = [[0.55, -0.45], [-0.45, 0.55]]
coordinate_mu = [0.5, 0.5]

# for coordinateDistribution == 'GP'
covFun = 'squaredExponential'
cov_l = 0.1
sig_scale = 1.5
sigmaGP_r = 0.4
lengthScale_r = .05

radiiDistribution = 'lognGP'
# to avoid circles on boundaries. Min. distance of circle centers to (lo., r., u., le.) boundary
margins = (0.003, 0.003, 0.003, 0.003)
r_params = (-5.53, .3)

# Flow boundary condition for velocity on domain boundary
u_x = '1.0-1.0*x[1]'
u_y = '2.0-1.0*x[0]'

flowField = df.Expression((u_x, u_y), degree=2)
u_x = u_x.replace('*', '')
u_y = u_y.replace('*', '')

foldername = '/home/constantin/python/data/stokesEquation/meshSize=' + str(nElements)

if porousMedium == 'nonOverlappingCircles':
    foldername += '/nonOverlappingDisks/margins=' + str(margins[0]) + '_' + str(margins[1]) + '_' + str(margins[2]) + \
                  '_' + str(margins[3]) + '/N~' + nExclusionsDist

    if nExclusionsDist == 'logn':
        foldername += '/mu=' + str(nExclusionParams[0]) + '/sigma=' + str(nExclusionParams[1])
    else:
        raise Exception('Invalid number of exclusions distribution')

    foldername += '/x~' + coordinateDistribution
    if coordinateDistribution == 'gauss':
        foldername += '/mu=' + str(coordinate_mu[0]) + '_' + str(coordinate_mu[1]) + \
                      '/cov=' + str(coordinate_cov[0][0]) + '_' + str(coordinate_cov[0][1]) + '_' + \
                      str(coordinate_cov[1][1])
    elif coordinateDistribution == 'GP':
        foldername += '/cov=' + covFun + '/l=' + str(cov_l) + '/sig_scale=' + str(sig_scale)
    elif coordinateDistribution == 'engineered' or coordinateDistribution == 'tiles':
        pass
    else:
        raise Exception('Invalid coordinates distribution')

    foldername += '/r~' + radiiDistribution
    if radiiDistribution == 'lognGP':
        foldername += '/mu=' + str(r_params[0]) + '/sigma=' + str(r_params[1]) + \
                      '/sigmaGP_r=' + str(sigmaGP_r) + '/l=' + str(lengthScale_r)
    else:
        foldername += '/mu=' + str(r_params[0]) + '/sigma=' + str(r_params[1])

# Set external boundaries of domain
class DomainBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS \
               or x[0] > (1.0 - df.DOLFIN_EPS) or x[0] < df.DOLFIN_EPS


# Initialize sub-domain instances for outer domain boundaries
domainBoundary = DomainBoundary()

for meshNumber in meshes:
    # set up file names
    # meshfile = foldername + '/mesh' + str(meshNumber) + '.xml'
    meshfile = foldername + '/mesh' + str(meshNumber) + '.mat'
    solutionfolder = foldername + '/p_bc=0.0' + '/u_x=' + u_x + '_u_y=' + u_y
    if not os.path.exists(solutionfolder):
        os.makedirs(solutionfolder)
    solutionfile = solutionfolder + '/solution' + str(meshNumber) + '.mat'

    # create computation_started.txt if not existent
    if not os.path.isfile(solutionfolder + '/computation_started.txt'):
        started_file = open(solutionfolder + '/computation_started.txt', 'w')
        started_file.close()

    started_file = open(solutionfolder + '/computation_started.txt', 'r')
    started_computations = started_file.readlines()
    started_file.close()
    print('started_computations == ', started_computations)
    while ((not os.path.isfile(meshfile)) or os.path.isfile(solutionfile)
           or ((str(meshNumber) + '\n') in started_computations)) and meshNumber < meshes[-1]:
        # print('Mesh ', str(meshNumber), ' does not exist or solution already computed. Passing to next mesh...')
        # print('mesh path == ', meshfile)
        # print('solution path == ', solutionfile)
        meshNumber += 1
        # meshfile = foldername + '/mesh' + str(meshNumber) + '.xml'
        meshfile = foldername + '/mesh' + str(meshNumber) + '.mat'
        solutionfile = solutionfolder + '/solution' + str(meshNumber) + '.mat'

    # write mesh number to file s.t. it is clear that solution is currently computed
    with open(solutionfolder + '/computation_started.txt', 'a') as started_file:
        started_file.write(str(meshNumber) + '\n')

    print('Loading mesh ', str(meshNumber), '...')
    # outdated
    # mesh = df.Mesh(foldername + '/mesh' + str(meshNumber) + '.xml')

    # load mesh from mat file
    mesh_data = sio.loadmat(foldername + '/mesh' + str(meshNumber) + '.mat')
    x = mesh_data['x']
    cells = mesh_data['cells']
    cells -= 1.0 # matlab to python indexing
    cells = np.array(cells, dtype=np.uintp)
    editor = df.MeshEditor()
    mesh = df.Mesh()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(x.shape[0])
    editor.init_cells(cells.shape[0])
    for k, point in enumerate(x):
        editor.add_vertex(k, point[:2])
    for k, cell in enumerate(cells):
        editor.add_cell(k, cell)
    editor.close()
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
    f = df.Constant((0.0, 0.0))  # right hand side
    a = mu * df.inner(df.grad(u), df.grad(v)) * df.dx + df.div(v) * p * df.dx + q * df.div(u) * df.dx
    L = df.inner(f, v) * df.dx

    # Form for use in constructing preconditioner matrix
    b = df.inner(df.grad(u), df.grad(v)) * df.dx + p * q * df.dx

    # Assemble system
    A, bb = df.assemble_system(a, L, bcs)

    # Assemble preconditioner system
    P, btmp = df.assemble_system(b, L, bcs)

    # Create Krylov solver and AMG preconditioner
    solver = df.KrylovSolver(krylov_method, 'amg')

    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    # Solve
    print('Solving PDE...')
    t = time.time()
    U = df.Function(W)

    try:
        solver.solve(U.vector(), bb)
        elapsed_time = time.time() - t
        print('PDE solved. Time: ', elapsed_time)
        print('sample: ', meshNumber)

        # Get sub-functions
        u, p = U.split()

        sio.savemat(solutionfile, {'u': np.reshape(u.compute_vertex_values(), (2, -1)), 'p': p.compute_vertex_values(),
                                   'x': mesh.coordinates()})

    except:
        print('Solver failed to converge. Passing to next mesh...')


