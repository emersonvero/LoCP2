import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

alpha = 100


x, y = np.mgrid[-1:1:.01, -1:1:.01]

pos = np.dstack((x, y))
zz = multivariate_normal.pdf(pos,mean=None,cov=np.eye(2)/alpha)

print(zz.shape)

fig=plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x,y,zz,cmap='viridis')
plt.show()