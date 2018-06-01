"""Module containing all the feature functions"""

import numpy as np
import math
import dolfin as df


def volumeFractionCircExclusions(diskCenters, diskRadii, meshRF):
    # diskCenters:      numpy vector of disk exclusion center coordinates
    # diskRadii:        numpy vector of disk exclusion radii
    # mesh:             coarse random field mesh object

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


