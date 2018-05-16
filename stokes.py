"""This should be the main module for the python Stokes/Darcy problem.
The code architecture still needs to be specified."""

import numpy as np
import dolfin as df
import os
import scipy.io as sio
import time

class StokesData():
    # Properties
    # All data parameters specified here

    folderbase = '/home/constantin'

    # general parameters
    meshes = np.arange(0, 4)                       # vector of random meshes to load
    porousMedium = 'nonOverlappingDisks'            # circles or randomField
    nElements = 128                                 #

    # physical parameters
    viscosity = 1.0

    # microstructure parameters
    nExclusionsDist = 'logn'                        # number of exclusions distribution
    nExclusionParams = (5.5, 1.0)                   # for logn: mu and sigma of logn dist.
    coordDist = 'gauss'                             # distribution of circ. exclusions in space
    coord_mu = [.7, .3]
    coord_cov = [[0.2, 0.0], [0.0, 0.3]]            # covariance of spatial disk distribution
    radiiDist = 'logn'                              # dist. of disk radii
    rParams = (-4.5, 0.7)                           # mu and sigma of disk radii distribution
    margins = (0.01, 0.01, 0.01, 0.01)              # margins of exclusions to boundaries

    # boundary conditions, specified as dolfin expressions
    # Flow boundary condition for velocity on domain boundary (avoid spaces for proper file path)
    # should be of the form u = (a_x + a_xy y, a_y + a_xy x)
    u_x = '0.0-2.0*x[1]'
    u_y = '1.0-2.0*x[0]'
    flowField = df.Expression((u_x, u_y), degree=2)
    # Pressure boundary condition field
    p_bc = '0.0'
    pressureField = df.Expression(p_bc, degree=2)
    # Interior boundary condition
    interiorBCtype = 'noslip'
    bodyForce = df.Constant((0.0, 0.0))  # right hand side

    class FlowBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            # SET FLOW BOUNDARIES HERE;
            # pressure boundaries are the complementary boundary
            return x[1] > 1.0 - df.DOLFIN_EPS or (x[1] < df.DOLFIN_EPS and x[0] > df.DOLFIN_EPS) or \
                   x[0] > 1.0 - df.DOLFIN_EPS or (x[0] < df.DOLFIN_EPS and x[1] > df.DOLFIN_EPS)


    def __init__(self):
        # Constructor
        self.flowBoundary = self.FlowBoundary()
        self.setFineDataPath()
        return

    def setFineDataPath(self):
        # set up finescale data path
        self.foldername = self.folderbase + '/python/data/stokesEquation/meshSize=' + str(self.nElements) + '/' \
                     + self.porousMedium + '/margins=' + str(self.margins[0]) + '_' + str(self.margins[1]) + '_' \
                     + str(self.margins[2]) + '_' + str(self.margins[3]) + '/N~' + self.nExclusionsDist
        if self.nExclusionsDist == 'logn':
            self.foldername += '/mu=' + str(self.nExclusionParams[0]) + '/sigma=' + str(self.nExclusionParams[1]) \
                          + '/x~' + self.coordDist

        if self.coordDist == 'gauss':
            self.foldername += '/mu=' + str(self.coord_mu[0]) + '_' + str(self.coord_mu[1]) + '/cov=' \
                          + str(self.coord_cov[0][0]) + '_' + str(self.coord_cov[0][1]) \
                          + '_' + str(self.coord_cov[1][1])

        self.foldername += '/r~' + self.radiiDist + '/mu=' + str(self.rParams[0]) + '/sigma=' + str(self.rParams[1])
        self.solutionfolder = self.foldername + '/p_bc=' + self.p_bc + '/u_x=' + self.u_x + '_u_y=' + self.u_y
        return

    def getFunctionSpace(self, mesh):
        # Define mixed function space (Taylor-Hood)
        u_e = df.VectorElement('CG', mesh.ufl_cell(), 2)
        p_e = df.FiniteElement('CG', mesh.ufl_cell(), 1)
        mixedEl = df.MixedElement([u_e, p_e])
        functionSpace = df.FunctionSpace(mesh, mixedEl)
        return functionSpace

    def getInteriorBC(self, functionSpace):
        # returns interior boundary bc for fenics

        # Define interior boundaries
        class InteriorBoundary(df.SubDomain):
            def inside(self, x, on_boundary):
                outerBoundary = x[1] > 1.0 - df.DOLFIN_EPS or x[1] < df.DOLFIN_EPS \
                                or x[0] > (1.0 - df.DOLFIN_EPS) or x[0] < df.DOLFIN_EPS
                return on_boundary and not outerBoundary

        # Initialize sub-domain instance for interior boundaries
        interiorBoundary = InteriorBoundary()

        # No-slip boundary condition for velocity on material interfaces
        if self.interiorBCtype == 'noslip':
            interiorBoundaryFlow = df.Constant((0.0, 0.0))
        else:
            raise ValueError('Unknown interior boundary condition.')

        # Boundary conditions for solid phase
        bc = df.DirichletBC(functionSpace.sub(0), interiorBoundaryFlow, interiorBoundary)
        return bc

    def getOuterBC(self, functionSpace):
        # BC's on outer domain boundary
        bc = df.DirichletBC(functionSpace.sub(0), self.flowField, self.flowBoundary)
        return bc

    def solvePDE(self, functionSpace, mesh, boundaryConditions):
        # Define variational problem
        (u, p) = df.TrialFunctions(functionSpace)
        (v, q) = df.TestFunctions(functionSpace)
        # get normal vectors
        n = df.FacetNormal(mesh)
        a = self.viscosity * df.inner(df.grad(u), df.grad(v)) * df.dx + df.div(v) * p * df.dx + q * df.div(u) * df.dx
        L = df.inner(self.bodyForce, v) * df.dx + self.pressureField * df.inner(n, v) * df.ds

        # Form for use in constructing preconditioner matrix
        b = df.inner(df.grad(u), df.grad(v)) * df.dx + p * q * df.dx

        # Assemble system
        A, bb = df.assemble_system(a, L, boundaryConditions)

        # Assemble preconditioner system
        P, btmp = df.assemble_system(b, L, boundaryConditions)

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

        # Create Krylov solver and AMG preconditioner
        solver = df.KrylovSolver(krylov_method)

        # Associate operator (A) and preconditioner matrix (P)
        solver.set_operators(A, P)

        # Solve
        U = df.Function(functionSpace)
        solver.solve(U.vector(), bb)
        return U

    def loadMesh(self, meshNumber):
        # load mesh from file
        mesh = df.Mesh(self.foldername + '/mesh' + str(meshNumber) + '.xml')
        return mesh

    def loadSolution(self, meshNumber):
        mesh = self.loadMesh(meshNumber)
        functionSpace = self.getFunctionSpace(mesh)
        solution = df.Function(functionSpace, self.solutionfolder + '/solution' + str(meshNumber) + '.xml')
        return solution, mesh, functionSpace

    def saveSolution(self, U, mesh, meshNumber, type='matlab'):
        # Save solution to mat file for easy read-in in matlab
        if not os.path.exists(self.solutionfolder):
            os.makedirs(self.solutionfolder)

        # Get sub-functions
        u, p = U.split()
        if type == 'matlab':
            sio.savemat(self.solutionfolder + '/solution' + str(meshNumber) + '.mat',
                    {'u': np.reshape(u.compute_vertex_values(), (2, -1)), 'p': p.compute_vertex_values(),
                     'x': mesh.coordinates()})
        elif type == 'python':
            solutionFile = df.File(self.solutionfolder + '/solution' + str(meshNumber) + '.xml')
            solutionFile << U
        else:
            raise ValueError('Unknown file format.')
        return

    def genData(self):
        for meshNumber in self.meshes:
            print('Current mesh number = ', meshNumber)
            mesh = self.loadMesh(meshNumber)
            functionSpace = self.getFunctionSpace(mesh)
            interiorBC = self.getInteriorBC(functionSpace)
            outerBC = self.getOuterBC(functionSpace)
            boundaryConditions = [interiorBC, outerBC]
            t = time.time()
            try:
                solution = self.solvePDE(functionSpace, mesh, boundaryConditions)
                self.saveSolution(solution, mesh, meshNumber, 'python')
                elapsed_time = time.time() - t
                print('equation system solved. Time: ', elapsed_time)
            except:
                print('Solver failed to converge. Passing to next mesh')
        return



# tests
stokesData = StokesData()
#stokesData.genData()
t = time.time()
solution = stokesData.loadSolution(1)
elapsed_time = time.time() - t
print('Solution loaded. Time: ', elapsed_time)

