

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import time
import multiprocessing
from flowproblem import FlowProblem
import featurefunctions as ff
import dolfin as df
import warnings
import os


class StokesData(FlowProblem):
    # Properties

    folderbase = '/home/constantin/cluster'

    # All data parameters specified here
    medium = 'nonOverlappingDisks'  # circles or randomField

    # physical parameters
    viscosity = 1.0

    # general parameters
    samples = np.arange(0, 2048)  # list of random meshes to load
    nElements = 256

    # microstructure parameters
    nExclusionsDist = 'logn'  # number of exclusions distribution
    nExclusionParams = (7.2, 0.6)  # for logn: mu and sigma of logn dist.
    coordDist = 'gauss_randmu'  # distribution of circ. exclusions in space
    coord_mu = [.8, .8]
    coord_cov = [[0.55, -0.45], [-0.45, 0.55]]  # covariance of spatial disk distribution
    radiiDist = 'logn'  # dist. of disk radii
    rParams = (-5.53, 0.3)  # mu and sigma of disk radii distribution
    margins = (0.002, 0.002, 0.002, 0.002)  # margins of exclusions to boundaries
    interiorBCtype = 'noslip'  # Boundary condition on exclusion boundary

    # data storage
    mesh = []
    solution = []
    p_interp = []
    v_interp = []

    designMatrix = [None] * samples.size

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
        elif self.coordDist == 'gauss_randmu':
            self.foldername += '/mu=rand' + '/cov=' + str(self.coord_cov[0][0]) + '_' +\
                               str(self.coord_cov[0][1]) + '_' + str(self.coord_cov[1][1]) + '/'

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
        bc = df.DirichletBC(functionSpace.sub(0), interiorBoundaryFlow, interiorBoundary,
                             method='topological', check_midpoint=False)
        return bc

    def getOuterBC(self, functionSpace):
        # BC's on outer domain boundary
        bc = df.DirichletBC(functionSpace.sub(0), self.flowField, self.flowBoundary, method='pointwise')
        return bc

    def solvePDE(self, functionSpace, mesh, boundaryConditions):
        # Define variational problem
        (u, p) = df.TrialFunctions(functionSpace)
        (v, q) = df.TestFunctions(functionSpace)
        # get normal vectors
        n = df.FacetNormal(mesh)
        a = self.viscosity * df.inner(df.grad(u), df.grad(v)) * df.dx + df.div(v) * p * df.dx + q * df.div(u) * df.dx
        # L = df.inner(self.bodyForce, v) * df.dx + self.pressureField * df.inner(n, v) * df.ds
        L = df.inner(self.bodyForce, v) * df.dx + df.inner(df.dot(self.stressField, n), v) * df.ds

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
        solver = df.KrylovSolver(krylov_method, "amg")
        solver.parameters["relative_tolerance"] = 1e-12
        solver.parameters["absolute_tolerance"] = 1e-12
        # solver.parameters["monitor_convergence"] = True

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

    def loadMicrostructureInformation(self, meshNumber):
        # Loads centers and radii of microstructure exclusions
        microstruct = sio.loadmat(self.foldername + '/microstructureInformation' + str(meshNumber) + '.mat')
        return microstruct

    def saveSolution(self, solutionFunction, meshNumber, filetype='numpy_compressed'):
        mesh = solutionFunction.function_space().mesh()

        if not os.path.exists(self.solutionfolder):
            os.makedirs(self.solutionfolder)

        if filetype == 'hdf5':
            hdf = df.HDF5File(mesh.mpi_comm(), self.solutionfolder + '/solution' + str(meshNumber) + '.h5', "w")
            hdf.write(solutionFunction, 'solution')
            hdf.close()
        elif filetype == 'matlab':
            v, p = solutionFunction.split()
            sio.savemat(self.solutionfolder + '/solution' + str(meshNumber) + '.mat',
                        {'u': np.reshape(v.compute_vertex_values(), (2, -1)), 'p': p.compute_vertex_values(),
                         'x': mesh.coordinates()})
        elif filetype == 'numpy':
            U = solutionFunction.vector().get_local()
            np.save(self.solutionfolder + '/solution' + str(meshNumber), U)
        elif filetype == 'numpy_compressed':
            U = solutionFunction.vector().get_local()
            np.savez_compressed(self.solutionfolder + '/solution' + str(meshNumber), U)

    def loadSolution(self, meshNumber, filetype='numpy_compressed'):
        mesh = self.loadMesh(meshNumber)

        if filetype == 'hdf5':
            hdf = df.HDF5File(df.mpi_comm_world(), self.solutionfolder + '/solution' + str(meshNumber) + '.h5', "r")

            functionSpace = getFunctionSpace(mesh)
            solutionFunction = df.Function(functionSpace)
            hdf.read(solutionFunction, 'solution')
            hdf.close()
        elif filetype == 'numpy':
            functionSpace = getFunctionSpace(mesh)
            solutionFunction = df.Function(functionSpace)
            U = np.load(self.solutionfolder + '/solution' + str(meshNumber) + '.npy')
            solutionFunction.vector().set_local(U)
        elif filetype == 'numpy_compressed':
            functionSpace = getFunctionSpace(mesh)
            solutionFunction = df.Function(functionSpace)
            U = np.load(self.solutionfolder + '/solution' + str(meshNumber) + '.npz')
            solutionFunction.vector().set_local(U['arr_0'])
        else:
            print('Unknown file format')

        return solutionFunction, mesh

    def genData(self):
        for meshNumber in self.samples:
            print('Current mesh number = ', meshNumber)
            mesh = self.loadMesh(meshNumber)
            functionSpace = getFunctionSpace(mesh)
            interiorBC = self.getInteriorBC(functionSpace)
            outerBC = self.getOuterBC(functionSpace)
            boundaryConditions = [interiorBC, outerBC]
            t = time.time()
            # try:
            solution = self.solvePDE(functionSpace, mesh, boundaryConditions)
            # self.saveSolution(solution, meshNumber)
            self.saveSolution(solution, meshNumber, 'matlab')
            elapsed_time = time.time() - t
            print('Stokes PDE solved. Time: ', elapsed_time)
            # except:
            #     print('Solver failed to converge. Passing to next mesh')
        return

    def loadData(self, quantities):
        # load and save data to object property
        # quantities: list of quantities specified by strings: 'mesh', 'pressure', 'velocity'
        for meshNumber in self.samples:
            if 'mesh' in quantities:
                self.mesh.append(self.loadMesh(meshNumber))
            if 'solution' in quantities:
                U, _ = self.loadSolution(meshNumber, 'hdf5')
                self.solution.append(U)

    def shiftData(self, point=np.array([.0, .0]), value=.0, quantity='pressure'):
        # Shifts solution to value 'value' at point 'point'

        p_old = np.empty(1, dtype=float)
        if quantity == 'pressure':
            print('Shifting data such that pressure is ', value, ' at ', point, ' ...')
            for p in self.p_interp:
                p.eval(p_old, point)
                shift_value = value - p_old
                p.vector().set_local(p.vector().get_local() + shift_value)
                # val = np.empty(1, dtype=float)
                # p.eval(val, np.array([.0, .0]))
                # print('p_interp origin = ', val)
            print('...data shifted.')
        else:
            raise ValueError('shiftData only implemented for pressure field')

    def interpolate(self, quantities, modelParameters):
        # interpolation of fine scale solution to a regular mesh
        # So far only implemented for pressure

        print('Interpolating data onto regular grid...')
        for solution_n in self.solution:

            if 'p' in quantities:
                _, p_n = solution_n.split()
                p_n.set_allow_extrapolation(True)
                p_int_n = df.interpolate(p_n, modelParameters.pInterpSpace)
                self.p_interp.append(p_int_n)

        print('...data interpolated.')

    def evaluateFeatures(self, Phi, sample_index, modelParameters, writeTextFile=False):
        # Evaluate feature functions and set up design matrix for sample n

        Phi[sample_index] = np.empty((modelParameters.coarseMesh.num_cells(), 0))

        file = open('./data/featureFunctions.txt', 'w') if writeTextFile else False

        # Centers and radii of exclusions
        microstructureInformation = self.loadMicrostructureInformation(sample_index)

        # Constant 1
        Phi[sample_index] = np.append(Phi[sample_index],
                          np.ones((modelParameters.coarseMesh.num_cells(), 1)), axis=1)
        file.write('constant\n') if writeTextFile else False

        # Pore fraction
        Phi[sample_index] =\
            np.append(Phi[sample_index], ff.volumeFractionCircExclusions(self.mesh[sample_index],
                                                                 modelParameters.coarseMesh), axis=1)
        file.write('poreFraction\n') if writeTextFile else False

        # Radii moments
        Phi[sample_index] = \
            np.append(Phi[sample_index], ff.momentOfRadii(microstructureInformation, modelParameters.coarseMesh,
                                                            moment=1.0), axis=1)
        file.write('radius moment 1.0\n') if writeTextFile else False

        Phi[sample_index] = \
            np.append(Phi[sample_index], ff.momentOfRadii(microstructureInformation, modelParameters.coarseMesh,
                                                            moment=2.0), axis=1)
        file.write('radius moment 2.0\n') if writeTextFile else False

        # Specific surface
        Phi[sample_index] = \
            np.append(Phi[sample_index], ff.specificSurface(microstructureInformation, self.mesh[sample_index],
                                                            modelParameters.coarseMesh), axis=1)
        file.write('specific surface\n') if writeTextFile else False

        # e_V, h_V
        e_V, h_V = ff.voidNearestSurfaceExclusion(microstructureInformation, self.mesh[sample_index],
                                                  modelParameters.coarseMesh, .08)
        Phi[sample_index] = np.append(Phi[sample_index], e_V, axis=1)
        file.write('e_V08\n') if writeTextFile else False

        Phi[sample_index] = np.append(Phi[sample_index], h_V, axis=1)
        file.write('h_V08\n') if writeTextFile else False

        e_V, h_V = ff.voidNearestSurfaceExclusion(microstructureInformation, self.mesh[sample_index],
                                                  modelParameters.coarseMesh, .04)
        Phi[sample_index] = np.append(Phi[sample_index], e_V, axis=1)
        file.write('e_V04\n') if writeTextFile else False

        Phi[sample_index] = np.append(Phi[sample_index], h_V, axis=1)
        file.write('h_V04\n') if writeTextFile else False

        e_V, h_V = ff.voidNearestSurfaceExclusion(microstructureInformation, self.mesh[sample_index],
                                                  modelParameters.coarseMesh, .02)
        Phi[sample_index] = np.append(Phi[sample_index], e_V, axis=1)
        file.write('e_V02\n') if writeTextFile else False

        Phi[sample_index] = np.append(Phi[sample_index], h_V, axis=1)
        file.write('h_V02\n') if writeTextFile else False

        e_V, h_V = ff.voidNearestSurfaceExclusion(microstructureInformation, self.mesh[sample_index],
                                                  modelParameters.coarseMesh, .01)
        Phi[sample_index] = np.append(Phi[sample_index], e_V, axis=1)
        file.write('e_V01\n') if writeTextFile else False

        Phi[sample_index] = np.append(Phi[sample_index], h_V, axis=1)
        file.write('h_V01\n') if writeTextFile else False

        # Lineal path for pore phase (True)
        Phi[sample_index] = \
            np.append(Phi[sample_index], ff.matrixLinealPath(microstructureInformation, self.mesh[sample_index],
                                                             modelParameters.coarseMesh, .04), axis=1)
        file.write('linealPath04\n') if writeTextFile else False

        Phi[sample_index] = \
            np.append(Phi[sample_index], ff.matrixLinealPath(microstructureInformation, self.mesh[sample_index],
                                                             modelParameters.coarseMesh, .02), axis=1)
        file.write('linealPath02\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.matrixLinealPath(microstructureInformation, self.mesh[sample_index],
        #                                                      modelParameters.coarseMesh, .01), axis=1)
        # file.write('linealPath01\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.matrixLinealPath(microstructureInformation, self.mesh[sample_index],
        #                                                      modelParameters.coarseMesh, .005), axis=1)
        # file.write('linealPath005\n') if writeTextFile else False
        #
        # # Chord length density
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.chordLengthDensity(microstructureInformation, self.mesh[sample_index],
        #                                                        modelParameters.coarseMesh, .04), axis=1)
        # file.write('ChordLengthDensity04\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.chordLengthDensity(microstructureInformation, self.mesh[sample_index],
        #                                                        modelParameters.coarseMesh, .02), axis=1)
        # file.write('ChordLengthDensity02\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.chordLengthDensity(microstructureInformation, self.mesh[sample_index],
        #                                                        modelParameters.coarseMesh, .01), axis=1)
        # file.write('ChordLengthDensity01\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.chordLengthDensity(microstructureInformation, self.mesh[sample_index],
        #                                                        modelParameters.coarseMesh, .005), axis=1)
        # file.write('ChordLengthDensity005\n') if writeTextFile else False
        #
        # # mean chord length
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.meanChordLength(microstructureInformation, self.mesh[sample_index],
        #                                                     modelParameters.coarseMesh), axis=1)
        # file.write('MeanChordLength\n') if writeTextFile else False
        #
        # # disk distance
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.diskDistance(microstructureInformation, modelParameters.coarseMesh,
        #                                                  'mean', 'edge2edge'), axis=1)
        # file.write('meanEdge2EdgeDistance\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.diskDistance(microstructureInformation, modelParameters.coarseMesh,
        #                                                  'mean', 2.0), axis=1)
        # file.write('meanDistance\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.diskDistance(microstructureInformation, modelParameters.coarseMesh,
        #                                                  'mean', 1.0), axis=1)
        # file.write('mean1Distance\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.diskDistance(microstructureInformation, modelParameters.coarseMesh,
        #                                                  'mean', -1.0), axis=1)
        # file.write('mean-1Distance\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.diskDistance(microstructureInformation, modelParameters.coarseMesh,
        #                                                  'max', 2.0), axis=1)
        # file.write('maxDistance\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.diskDistance(microstructureInformation, modelParameters.coarseMesh,
        #                                                  'min', 2.0), axis=1)
        # file.write('minDistance\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.diskDistance(microstructureInformation, modelParameters.coarseMesh,
        #                                                  'std', 2.0), axis=1)
        # file.write('stdDistance\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.diskDistance(microstructureInformation, modelParameters.coarseMesh,
        #                                                  'var', 2.0), axis=1)
        # file.write('varDistance\n') if writeTextFile else False
        #
        # # 2-point correlation for pore phase (True)
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.twoPointCorrelation(microstructureInformation,
        #                                                         modelParameters.coarseMesh, True, .04), axis=1)
        # file.write('pore2-point-correlation04\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.twoPointCorrelation(microstructureInformation,
        #                                                         modelParameters.coarseMesh, True, .02), axis=1)
        # file.write('pore2-point-correlation02\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.twoPointCorrelation(microstructureInformation,
        #                                                         modelParameters.coarseMesh, True, .01), axis=1)
        # file.write('pore2-point-correlation01\n') if writeTextFile else False
        #
        # Phi[sample_index] = \
        #     np.append(Phi[sample_index], ff.twoPointCorrelation(microstructureInformation,
        #                                                         modelParameters.coarseMesh, True, .005), axis=1)
        # file.write('pore2-point-correlation005\n') if writeTextFile else False

        file.close() if writeTextFile else False

    def computeFeatureFunctionMinMax(self):
        # Computes min / max of feature function outputs over training data, separately for every macro cell
        featFunMin = self.designMatrix[0].copy()
        featFunMax = self.designMatrix[0].copy()
        for Phi in self.designMatrix:
            featFunMin[featFunMin > Phi] = Phi[featFunMin > Phi].copy()
            featFunMax[featFunMax < Phi] = Phi[featFunMax < Phi].copy()

        return featFunMin, featFunMax

    def rescaleDesignMatrix(self, modelParams):
        # Rescale design matrix to have feature function outputs between 0 and 1

        print('Rescaling design matrix...')

        if modelParams.featFunMin is None or modelParams.featFunMax is None:
            modelParams.featFunMin, modelParams.featFunMax = self.computeFeatureFunctionMinMax()

        featFunDiff = modelParams.featFunMax - modelParams.featFunMin
        # To avoid irregularities due to rescaling (if every macro cell has the same feature function output).
        # Like this, rescaling does not have any effect
        modelParams.featFunMin[abs(featFunDiff) < np.finfo(float).eps] = 0.0
        featFunDiff[abs(featFunDiff) < np.finfo(float).eps] = 1.0

        for n in range(0, self.samples.size):
            self.designMatrix[n] -= modelParams.featFunMin
            self.designMatrix[n] /= featFunDiff

        print('...design matrix rescaled.')

    def normalizeDesignMatrix(self, modelParams):
        if modelParams.normalization == 'rescale':
            self.rescaleDesignMatrix(modelParams)
        else:
            warnings.warn('Unknown design matrix normalization. No normalization is performed.')

    def shapeToLocalDesignMatrix(self, sparse=False):
        # Reshape design matrix in such a way that it is suitable for local theta_c's
        print('Reshaping design matrix for separate feature coefficients theta_c',
              ' for each macro-cell in a microstructure...')

        nElc, nFeatureFunctions = self.designMatrix[0].shape

        for n in range(0, self.samples.size):
            Phi_temp = np.zeros((nElc, nElc * nFeatureFunctions))
            for k in range(0, nElc):
                Phi_temp[k, (k*nFeatureFunctions):((k + 1)*nFeatureFunctions)] = \
                    self.designMatrix[n][k, :].copy()
            self.designMatrix[n] = Phi_temp.copy()
            if sparse:
                self.designMatrix[n] = sp.csr_matrix(self.designMatrix[n])

        print('...design matrices reshaped to local.')

    def computeDesignMatrix(self, modelParameters, parallel_or_serial='parallel'):
        # Evaluate features for all samples and write to design matrix property

        print('Evaluate features ...')
        if parallel_or_serial == 'parallel':

            if __name__ == 'stokesdata':

                manager = multiprocessing.Manager()
                Phi_temp = manager.dict()
                # Set up processes
                processes = []
                # to write feature text file
                processes.append(multiprocessing.Process(
                    target=self.evaluateFeatures, args=(Phi_temp, 0, modelParameters, True)))
                for n in range(1, self.samples.size):
                    processes.append(multiprocessing.Process(
                        target=self.evaluateFeatures, args=(Phi_temp, n, modelParameters, False)))

                # Start processes
                for process in processes:
                    process.start()

                # Join processes
                for process in processes:
                    process.join()

                # convert to regular list
                self.designMatrix = self.samples.size * [None]
                for n in range(0, self.samples.size):
                    self.designMatrix[n] = Phi_temp[n].copy()

        else:
            self.designMatrix = self.samples.size * [None]
            self.evaluateFeatures(self.designMatrix, 0, modelParameters, True)   # to write feature text file
            for n in range(1, self.samples.size):
                self.evaluateFeatures(self.designMatrix, n, modelParameters, False)
        print('...features evaluated.')

# Static functions
def getFunctionSpace(mesh):
    # Define mixed function space (Taylor-Hood)
    u_e = df.VectorElement('CG', mesh.ufl_cell(), 2)
    p_e = df.FiniteElement('CG', mesh.ufl_cell(), 1)
    mixedEl = df.MixedElement([u_e, p_e])
    functionSpace = df.FunctionSpace(mesh, mixedEl)
    return functionSpace




