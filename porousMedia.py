"""This should be a porous media library only. No specifications
to any PDE here!"""

import numpy as np
import mshr
import dolfin as df
from skimage import measure
import time

#Methods

def findPolygones(image, isovalue, showImg=False):
    """Finds the vertices of polygones encircling solid phase blobs"""

    print('Finding polygones with marching squares...')
    contours = measure.find_contours(image, isovalue,
                                     positive_orientation='high', fully_connected='low')
    print('done.')

    #  Show image with polygones
    if showImg:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image >= isovalue, cmap=plt.cm.inferno)
        for i in range(0, len(contours)):
            contour = contours[i]
            plt.plot(contour[:, 1], contour[:, 0])
        plt.show()

    return contours

def generateMesh(domain, discretization=128):
    """Generates a finite element mesh using mshr based on an indicator function
    specifying the domain"""

    print('generating FE mesh...')
    t = time.time()
    mesh = mshr.generate_mesh(domain, discretization)
    elapsed_time = time.time() - t
    print('done. Time: ', elapsed_time)

    return mesh


def discretizeRandField(randField, domain_a=(0.0, 0.0), domain_b=(1.0, 1.0), nDiscretize=(128, 128)):
    """Generate pixel image from continuous indicator function
    specifying porous domain"""

    print('Discretizing random field...')
    x = np.linspace(domain_a[0], domain_b[0], nDiscretize[0])
    y = np.linspace(domain_a[1], domain_b[1], nDiscretize[1])
    discretizedRandField = np.zeros(nDiscretize)
    for i in range(0, nDiscretize[0]):
        for j in range(0, nDiscretize[1]):
            discretizedRandField[i, j] = randField(np.array([x[i], y[j]]))
    print('done.')

    return discretizedRandField


def rescalePolygones(contours, domain_a=(0.0, 0.0), domain_b=(1.0, 1.0), nDiscretize=(128, 128)):
    """Rescales blob polygon contours to fit in domain"""

    print('Rescaling blob contours to fit domain...')
    for blob in range(0, len(contours)):
        contour = contours[blob]
        contour[:, 0] = ((domain_b[0] - domain_a[0])/(nDiscretize[0] - 1)) * \
            contour[:, 0] + domain_a[0]
        contour[:, 1] = ((domain_b[1] - domain_a[1])/(nDiscretize[1] - 1)) * \
            contour[:, 1] + domain_a[1]
    print('done.')
    return contours


def substractPores(contours, domain_a=(0.0, 0.0), domain_b=(1.0, 1.0)):
    """Substracting polygones of pores of dolfin.domain"""

    domain = mshr.Rectangle(df.Point(domain_a[0], domain_a[1]),
                            df.Point(domain_b[0], domain_b[1]))

    print('Substracting blobs as polygones from domain...')
    for blob in range(0, len(contours)):
        contour = contours[blob]
        vertexList = []
        for i in range(0, contour.shape[0]):
            # Construct list of df.Point's for polygon vertices
            x = np.array([contour[i, 0], contour[i, 1]])
            vertexList.append(df.Point(np.squeeze(x)))
        # Substract polygon from domain
        try:
            mPolygon = mshr.Polygon(vertexList)
        except RuntimeError:
            print('Invalid blob at')
            print('blob = ', blob)
            print('contour = ', contour)
        domain -= mPolygon
    print('done.')

    return domain
