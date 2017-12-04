import dolfin as df
import matplotlib.pyplot as plt


mesh = df.Mesh("/home/constantin/python/data/stokesEquation/meshes/meshSize=128/"
               "randFieldDiscretization=64/cov=matern/params=[5.0]/l=0.008_0.008/mesh0.xml")
df.plot(mesh)
plt.show()
