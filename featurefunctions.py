"""Module containing all the feature functions"""

import numpy as np
import math
import dolfin as df
import time
import romtoolbox as tb


def volumeFractionCircExclusions(fineMesh, meshRF):
    # fineMesh:         finescale data mesh
    # mesh:             coarse random field mesh object

    '''
    A = np.empty(meshRF.num_cells())
    A0 = A
    circle_surfaces = math.pi * diskRadii**2
    circles_in_cll = np.zeros(diskRadii.shape, dtype=bool)
    for cll_idx in range(0, meshRF.num_cells()):
        cll = df.Cell(meshRF, cll_idx)
        A[cll_idx] = A0[cll_idx] = cll.volume()

        for circ_idx in range(0, diskRadii.size):
            center = df.Point(diskCenters[circ_idx, :])
            circles_in_cll[circ_idx] = cll.contains(center)     # bool array of exclusion centers in cell cll

        A[cll_idx] = A0[cll_idx] - np.sum(circle_surfaces[circles_in_cll])

    poreFraction = A/A0
    poreFraction[poreFraction <= 0] = df.DOLFIN_EPS     #can happen if circles lie on boundary
    '''

    poreFraction = np.empty((meshRF.num_cells(), 1))
    constFunSpace = df.FunctionSpace(meshRF, 'DG', 0)
    constFun = df.Function(constFunSpace)
    dx = df.Measure('dx', fineMesh)     # integrate over finescale data domain
    for cll_idx in range(0, meshRF.num_cells()):
        constFun.vector().set_local(np.zeros(meshRF.num_cells()))
        constFun.vector()[cll_idx] = 1.0    # gives indicator function for being inside of cell cll_idx

        poreVolume = df.assemble(constFun * dx)    # integ. indic. func. over finescale domain (exclusions respected)
        cll = df.Cell(meshRF, cll_idx)
        poreFraction[cll_idx] = poreVolume/cll.volume()

    return poreFraction


def momentOfRadii(microstructureInformation, meshRF, moment):
    # Computes the sum of moments of disk radii divided by cell area
    # diskCenters: clear
    # diskRadii: clear

    diskRadii = np.squeeze(microstructureInformation['diskRadii'], axis=0)
    diskCenters = microstructureInformation['diskCenters']

    mean_radii_moments = np.zeros((meshRF.num_cells(), 1))

    radii_samples = diskRadii**moment
    k = 0
    for cll_idx in range(meshRF.num_cells()):

        cll = df.Cell(meshRF, cll_idx)

        circles_in_n = np.empty_like(diskRadii, dtype=bool)
        i = 0
        for x in diskCenters:
            circles_in_n[i] = cll.contains(df.Point(np.array(x)))
            i += 1
        mean_radii_moments[k] = np.mean(radii_samples[circles_in_n])
        k += 1

    return mean_radii_moments


def specificSurface(microstructureInformation, fineMesh, meshRF):
    # Specific surface for non-overlapping polydisperse spheres
    # See Torquato 6.33

    meanRadii = momentOfRadii(microstructureInformation, meshRF, 1.0)
    meanSqRadii = momentOfRadii(microstructureInformation, meshRF, 2.0)

    porefrac = volumeFractionCircExclusions(fineMesh, meshRF)
    s = 2.0*(1.0 - porefrac)*(meanRadii/meanSqRadii)
    s[np.isnan(s)] = 0  # this occurs if macro - cell has no inclusions
    return s


def voidNearestSurfaceExclusion(microstructureInformation, fineMesh, meshRF, distance):
    # nearest surface functions for distance 'distance'. Torquato 6.50, 6.51

    porefrac = volumeFractionCircExclusions(fineMesh, meshRF)
    exclfrac = 1.0 - porefrac

    meanRadii = momentOfRadii(microstructureInformation, meshRF, 1.0)
    meanSqRadii = momentOfRadii(microstructureInformation, meshRF, 2.0)

    S = (meanRadii**2)/meanSqRadii
    a_0 = (1 + exclfrac*(S - 1))/(porefrac**2)
    a_1 = 1/porefrac

    x = distance/(2*meanRadii)
    e_v = porefrac* np.exp(-4*exclfrac*S*(a_0*x**2 + a_1*x))
    if np.any(np.logical_not(np.isfinite(e_v))):
        # print warning here?
        e_v[np.logical_not(np.isfinite(e_v))] = 1.0

    h_v = 2*((exclfrac*S)/meanRadii)*(2*a_0*x + a_1)*e_v
    if np.any(np.logical_not(np.isfinite(h_v))):
        h_v[np.logical_not(np.isfinite(h_v))] = 0.0

    return e_v, h_v


def matrixLinealPath(microstructureInformation, fineMesh, meshRF, distance):
    # Gives the lineal path function for the matrix phase according to the approximation in Torquato eq. 6.37

    meanRadii = momentOfRadii(microstructureInformation, meshRF, 1.0)
    meanSqRadii = momentOfRadii(microstructureInformation, meshRF, 2.0)

    porefrac = volumeFractionCircExclusions(fineMesh, meshRF)

    L = porefrac*np.exp(-(2*distance*(1 - porefrac)*meanRadii)/(np.pi*porefrac*meanSqRadii))
    L[np.logical_not(np.isfinite(L))] = 1.0 # Happens if there are no exclusions in macro - cell

    return L


def chordLengthDensity(microstructureInformation, fineMesh, meshRF, distance):
    # Mean chord length for non - overlapping polydisp.spheres according to Torquato 6.39

    meanRadii = momentOfRadii(microstructureInformation, meshRF, 1.0)
    meanSqRadii = momentOfRadii(microstructureInformation, meshRF, 2.0)

    porefrac = volumeFractionCircExclusions(fineMesh, meshRF)
    exclfrac = 1 - porefrac

    lc = .5*np.pi*(meanSqRadii/meanRadii)*(porefrac/exclfrac)
    if np.any(np.logical_not(np.isfinite(lc))):
        A = np.empty(meshRF.num_cells())
        for cll_idx in range(meshRF.num_cells()):
            cll = df.Cell(meshRF, cll_idx)
            A[cll_idx] = cll.volume()
        lc[np.logical_not(np.isfinite(lc))] = np.sqrt(A[np.logical_not(np.isfinite(lc))]) # set to ~ cell edge length

    cld = (1/lc)*np.exp(-(distance/lc))
    return cld


def meanChordLength(microstructureInformation, fineMesh, meshRF):
    # Mean chord length for non-overlapping polydisp. spheres according to Torquato 6.40

    meanRadii = momentOfRadii(microstructureInformation, meshRF, 1.0)
    meanSqRadii = momentOfRadii(microstructureInformation, meshRF, 2.0)

    porefrac = volumeFractionCircExclusions(fineMesh, meshRF)
    exclfrac = 1.0 - porefrac

    edg_max = meanRadii

    lc = .5*np.pi*(meanSqRadii/meanRadii)*(porefrac/exclfrac)
    # Set maximum chord length to .5
    lc[lc > .5] = .5
    # can happen in macro - cells with no exclusions
    lc[np.logical_not(np.isfinite(lc))] = .5

    return lc


def diskDistance(microstructureInformation, meshRF, property, p_norm):
    # Computes average/min/max/std distance of disks in a macro-cell

    diskRadii = np.squeeze(microstructureInformation['diskRadii'], axis=0)
    diskCenters = microstructureInformation['diskCenters']

    distQuantity = np.empty((meshRF.num_cells(), 1))
    for cll_idx in range(meshRF.num_cells()):
        cll = df.Cell(meshRF, cll_idx)
        exception_flag = False
        centers = np.array([]).reshape(0, 2)
        radii = np.array([])
        for excl_index in range(diskRadii.size):
            if cll.contains(df.Point(np.array(diskCenters[excl_index]))):
                centers = np.vstack((centers, np.array(diskCenters[excl_index])))
                radii = np.append(radii, diskRadii[excl_index])
        distances = np.empty(int(radii.size*(radii.size - 1)/2))

        ind = 0
        if radii.size > 1:   # we neeed at least 2 exclusions on macro-cell
            for i in range(radii.size):
                for j in range(i + 1, radii.size):
                    if p_norm == 'edge2edge':
                        # Computes 2 - norm, but from disk edge to disk edge
                        distances[ind] = np.linalg.norm(centers[i] - centers[j]) - radii[i] - radii[j]
                    else:
                        # Computes p - norm between disk centers
                        distances[ind] = np.linalg.norm(centers[i] - centers[j], p_norm)
                    ind += 1
        else:
            # warning('Cell with one or less exclusions. Setting distances = 0')
            distances = .0
            exception_flag = True

        if property == 'mean':
            distQuantity[cll_idx] = np.mean(distances)
        elif property == 'max':
            distQuantity[cll_idx] = np.max(distances)
        elif property == 'min':
            distQuantity[cll_idx] = np.min(distances)
        elif property == 'std':
            distQuantity[cll_idx] = np.std(distances)
        elif property == 'var':
            distQuantity[cll_idx] = np.var(distances)
        else:
            print('Unknown distance property')

        if exception_flag:
            # If there are no/one exclusions in macro - cell
            if radii.size == 1:
                distQuantity[cll_idx] = .5*cll.h()
            else:
                distQuantity[cll_idx] = cll.h()

            if property == 'var':
                distQuantity[cll_idx] = distQuantity[cll_idx]**2

    return distQuantity


def twoPointCorrelation(microstructureInformation, meshRF, phase, distance):

    diskRadii = np.squeeze(microstructureInformation['diskRadii'], axis=0)
    diskRadiiSq = diskRadii**2
    diskCenters = microstructureInformation['diskCenters']

    n = 0
    twoPtCorr = np.empty((meshRF.num_cells(), 1))
    t_init = time.time()
    for cll_idx in range(meshRF.num_cells()):
        cll = df.Cell(meshRF, cll_idx)
        converged = False
        nSamples = 1.0
        while not converged:
            # x1, x2 are random points in cell
            x1 = np.squeeze(tb.triangrnd(cll.get_vertex_coordinates()))

            # x2 is at distance 'distance' to x1 -- reject if x2 is not in cell
            x2 = np.array([np.Inf, np.Inf])
            t_in = time.time()
            t_elapsed = .0

            while (not cll.contains(df.Point(x2))) and (t_elapsed < .5):
                phi = 2*np.pi*np.random.rand()
                dx = distance * np.array([np.cos(phi), np.sin(phi)])
                x2 = x1 + dx
                t_elapsed = t_in - time.time()

            # Check if the points lie within a circle, i.e. outside domain
            isout1 = np.any(np.sum((x1 - diskCenters)**2, axis=1) < diskRadiiSq)
            isout2 = np.any(np.sum((x2 - diskCenters)**2, axis=1) < diskRadiiSq)

            # ~= because we want 'true' for pores and 'false' for solids
            if (isout1 != phase) and (isout2 != phase):
                # both points are in phase 'phase'
                twoPtCorr[cll_idx] = (1.0/nSamples)*((nSamples - 1.0)*twoPtCorr[cll_idx] + 1.0)
            else:
                twoPtCorr[cll_idx] = ((nSamples - 1)/nSamples)*twoPtCorr[cll_idx]

            mcerr_twoPtCorr = np.sqrt((twoPtCorr[cll_idx] - twoPtCorr[cll_idx]**2) / nSamples)
            # Error limits are hard-coded here
            t_elapsed_conv = t_init - time.time()

            if ((mcerr_twoPtCorr/(twoPtCorr[n] + 1e-12) < 0.05) and (nSamples > 50)) or (t_elapsed_conv > 10):
                converged = True

            nSamples += 1

    # Can this actually happen?
    twoPtCorr[twoPtCorr < .0] = 0

    return twoPtCorr

