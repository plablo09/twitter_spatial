import pyGDsandbox
from kde_maps import kde_grid
import numpy as np
import matplotlib.pyplot as plt
import pysal as ps
import glob
import ntpath



files = glob.glob('data/dia/*.shp')
n_files = len(files)
i = 0

# for file in files:
#     print ntpath.basename(file).split(".",1)[0]

for file in files:
    print "procesando archivo %s de %s" % (i,n_files)
    output_file = ntpath.basename(file).split(".",1)[0]
    points = np.array([point for point in ps.open(file)])
    #xyz, gdim = kde_grid(points, spaced=0.01)
    xyz, gdim, bw  = kde_grid(points, spaced=0.01)
    x = xyz[:, 0].reshape(gdim)
    y = xyz[:, 1].reshape(gdim)
    z = xyz[:, 2].reshape(gdim)

    levels = np.linspace(0, z.max(), 25)
    f = plt.figure(figsize=(8, 4))

    ax1 = f.add_subplot(121)
    ax1.contourf(x, y, z, levels=levels, cmap=plt.cm.bone)
    ax1.set_title("KDE")
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    ax2 = f.add_subplot(122)
    ax2.scatter(points[:, 0], points[:, 1])
    ax2.set_title("Original points")
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    plt.savefig('data/output_tiffs/'+ output_file + '.png')
    i+=1
