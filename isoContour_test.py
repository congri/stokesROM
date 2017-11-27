
import isoContour as iso
import numpy as np
from randomFieldGeneration import RandomField as rf
import scipy.io as sio
import matplotlib.pyplot as plt



ic = iso.IsoContour()

nMesh = 20
x = np.linspace(0, 1, nMesh)
X, Y = np.meshgrid(x, x)
img = 2*(X - 0.5)**2 + (Y - 0.5)**2


randomFieldObj = rf()
randomField = randomFieldObj.sample()
img = np.zeros([nMesh, nMesh])
for i in range(0, nMesh):
    for j in range(0, nMesh):
        img[i, j] = randomField(np.array([x[i], x[j]]))


sio.savemat('img.mat', {'img': img})
# temp = sio.loadmat('img.mat')
# img = temp['img']
obj, vert, lin = ic.isocontour(img, .8)


plt.show()