# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:32:51 2011

@author: endolith@gmail.com
"""

import numpy as np
import scipy.stats as stats
#from matplotlib.pyplot import imshow
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

# Create some dummy data
rvs = np.append(stats.norm.rvs(loc=2,scale=1,size=(200,1)),
                stats.norm.rvs(loc=1,scale=3,size=(200,1)),
                axis=1)

kde = stats.kde.gaussian_kde(rvs.T)

# Regular grid to evaluate kde upon
x_flat = np.r_[rvs[:,0].min():rvs[:,0].max():128j]
y_flat = np.r_[rvs[:,1].min():rvs[:,1].max():128j]
x,y = np.meshgrid(x_flat,y_flat)
grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

z = kde(grid_coords.T)
z = z.reshape(128,128)
f = plt.figure(figsize=(8, 4))
ax1 = f.add_subplot(121)
plt.scatter(rvs[:,0],rvs[:,1],alpha=0.5,color=u'white')
ax1.imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),origin=u'lower',extent=(rvs[:,0].min(),rvs[:,0].max(),rvs[:,1].min(),rvs[:,1].max()))
plt.show()
