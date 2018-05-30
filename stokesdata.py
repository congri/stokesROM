

import numpy as np
import scipy.io as sio
import time
from flowproblem import FlowProblem
from modelparameters import ModelParameters
import dolfin as df
import fenics_adjoint as dfa


class StokesData(FlowProblem, ModelParameters):
    # Properties

    folderbase = '/home/constantin/cluster'

    # All data parameters specified here
    medium = 'nonOverlappingDisks'  # circles or randomField

    # physical parameters
    viscosity = 1.0

    # general parameters
    meshes = np.arange(0, 1)  # vector of random meshes to load
    nElements = 128

    # microstructure parameters
    nExclusionsDist = 'logn'  # number of exclusions distribution
    nExclusionParams = (5.5, 1.0)  # for logn: mu and sigma of logn dist.
    coordDist = 'gauss'  # distribution of circ. exclusions in space
    coord_mu = [.7, .3]
    coord_cov = [[0.2, 0.0], [0.0, 0.3]]  # covariance of spatial disk distribution
    radiiDist = 'logn'  # dist. of disk radii
    rParams = (-4.5, 0.7)  # mu and sigma of disk radii distribution
    margins = (0.01, 0.01, 0.01, 0.01)  # margins of exclusions to boundaries
    interiorBCtype = 'noslip'  # Boundary condition on exclusion boundary

    # data storage
    mesh = []
    solution = []
    p_interp = []
    v_interp = []


    def __init__(self):
        # Constructor
        self.setFineDataPath()
        return

    def setFineDataPath(self):
        # set up finescale data path
        self.foldername = self.folderbase + '/python/data/stokesEquation/meshSize=' + str(self.nElements) + '/' \
                          + self.medium + '/margins=' + str(self.margins[0]) + '_' + str(self.margins[1]) + '_' \
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
        bc = dfa.DirichletBC(functionSpace.sub(0), interiorBoundaryFlow, interiorBoundary,
                             method='topological', check_midpoint=False)
        return bc

    def getOuterBC(self, functionSpace):
        # BC's on outer domain boundary
        bc = dfa.DirichletBC(functionSpace.sub(0), self.flowField, self.flowBoundary, method='pointwise')
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

    def saveSolution(self, solutionFunction, meshNumber, type='python'):
        mesh = solutionFunction.function_space().mesh()

        if type == 'python':
            hdf = df.HDF5File(mesh.mpi_comm(), self.solutionfolder + '/solution' + str(meshNumber) + '.h5', "w")
            hdf.write(solutionFunction, 'solution')
            hdf.close()
        elif type == 'matlab':
            v, p = solutionFunction.split()
            sio.savemat(self.solutionfolder + '/solution' + str(meshNumber) + '.mat',
                        {'u': np.reshape(v.compute_vertex_values(), (2, -1)), 'p': p.compute_vertex_values(),
                         'x': mesh.coordinates()})

    def loadSolution(self, meshNumber):
        mesh = self.loadMesh(meshNumber)

        hdf = df.HDF5File(df.mpi_comm_world(), self.solutionfolder + '/solution' + str(meshNumber) + '.h5', "r")

        functionSpace = getFunctionSpace(mesh)
        solution = dfa.Function(functionSpace)
        hdf.read(solution, 'solution')
        hdf.close()

        return solution, mesh

    def genData(self):
        for meshNumber in self.meshes:
            print('Current mesh number = ', meshNumber)
            mesh = self.loadMesh(meshNumber)
            functionSpace = getFunctionSpace(mesh)
            interiorBC = self.getInteriorBC(functionSpace)
            outerBC = self.getOuterBC(functionSpace)
            boundaryConditions = [interiorBC, outerBC]
            t = time.time()
            try:
                solution = self.solvePDE(functionSpace, mesh, boundaryConditions)
                self.saveSolution(solution, meshNumber)
                self.saveSolution(solution, meshNumber, 'matlab')
                elapsed_time = time.time() - t
                print('Stokes PDE solved. Time: ', elapsed_time)
            except:
                print('Solver failed to converge. Passing to next mesh')
        return

    def loadData(self, quantities):
        # load and save data to object property
        # quantities: list of quantities specified by strings: 'mesh', 'pressure', 'velocity'
        for meshNumber in self.meshes:
            if 'mesh' in quantities:
                self.mesh.append(self.loadMesh(meshNumber))
            if 'solution' in quantities:
                U, _ = self.loadSolution(meshNumber)
                self.solution.append(U)

    def interpolate(self, quantities):
        # interpolation of fine scale solution to a regular mesh
        # So far only implemented for pressure
        for solution_n in self.solution:

            if 'p' in quantities:
                _, p_n = solution_n.split()
                p_n.set_allow_extrapolation(True)
                self.p_interp.append(dfa.interpolate(p_n, self.pInterpSpace))


# Static functions
def getFunctionSpace(mesh):
    # Define mixed function space (Taylor-Hood)
    u_e = df.VectorElement('CG', mesh.ufl_cell(), 2)
    p_e = df.FiniteElement('CG', mesh.ufl_cell(), 1)
    mixedEl = df.MixedElement([u_e, p_e])
    functionSpace = df.FunctionSpace(mesh, mixedEl)
    return functionSpace
