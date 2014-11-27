from kde_maps import kde_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pysal as ps
import glob
import ntpath
from pysal.contrib.viz import mapping as maps


#sacamos los datos de los archivos

files = glob.glob('data/dia/*.shp')
def ordenalos(lista):
    orden =[]
    for l in lista:
        s = l.split("_",-1)[2]
        p = int(s.split(".")[0])
        orden.append((p,l))

    lista_ordenada = sorted(orden,key=lambda elemento: elemento[0])
    return lista_ordenada

lista_ordenada = ordenalos(files)
files_ordenados = []
for e in lista_ordenada:
    files_ordenados.append(e[1])


print files_ordenados



n_files = len(files_ordenados)
i = 0
xyz = []
gdim = []
bw = []
levels = []
for file in files_ordenados:
    print "procesando archivo %s de %s" % (i,n_files)
    print "nombre del archivo: " + file
    points = np.array([point for point in ps.open(file)])
    _xyz, _gdim, _bw  = kde_grid(points, spaced=0.01)
    _x = _xyz[:, 0].reshape(_gdim)
    _y = _xyz[:, 1].reshape(_gdim)
    _z = _xyz[:, 2].reshape(_gdim)
    xyz.append((_x,_y,_z))
    gdim.append(_gdim)
    bw.append(_bw)
    levels.append(np.linspace(0, _z.max(), 25))
    i+=1

#inicializamos el plot
fig = plt.figure()
ax = plt.axes(xlim=(460000,520000), ylim=(2100000,2210000))
#cont, = ax.contourf([], [], [], levels,cmap=plt.cm.bone)


def animate(i):
    x = xyz[i][0]
    y = xyz[i][1]
    z = xyz[i][2]
    cont = plt.contourf(x,y,z,levels[i],cmap=plt.cm.bone)

anim = animation.FuncAnimation(fig, animate, frames=n_files)
plt.show()
