"""generators-pf-rz.py -- Generating PFs using Risez energy method.

    We are running this script since it was extremely slow to run on notebook,
    even on hpcc.
"""

import sys
import os
import numpy as np
sys.path.append('../')
from vis.utils import transform as tr
from vis.generators import debmdk

dims = {"2d": 1000, "3d": 1500, '4d': 2500, '8d': 4100}

for dim in list(dims.keys())[-1:]:
    # Open all files
    fullpathf = "../data/debmdk-rz/{0:s}/dataf.csv".format(dim)
    path, filenamef = os.path.split(fullpathf)
    dirs = path.split('/')
    frontname = dirs[-2]
    os.makedirs(path, exist_ok=True)
    filenamex = filenamef.split('.')[0][0:-1] + 'x.csv'
    fullpathx = os.path.join(path, filenamex)
    print(path, filenamef, filenamex, frontname)

    np.random.seed(123456)
    r, n, m = 1, dims[dim], int(dim[0])
    F, X = debmdk.surface(r=r, n=n, m=m, mode='rz') # uniform
    print("F.shape:", F.shape)
    print("X.shape:", X.shape)
    
    Ip = tr.pfindices(F)
    F = F[Ip]
    X = X[Ip]
    print("F.shape:", F.shape)
    print("X.shape:", X.shape)

    print(F[0:3,:])
    print(X[0:3,:])
    
    # Just to make sure if we can get correct F from X
    F_ = np.zeros(F.shape)
    for i in range(m):
        if i < m-1:
            F_[:,i] = np.prod(np.sin(X[:,0:i] * (np.pi / 2)), axis=1) \
                    * np.cos(X[:,i] * (np.pi / 2))
        else:
            F_[:,i] = np.prod(np.sin(X[:,0:i-1] * (np.pi / 2)), axis=1) \
                    * np.sin(X[:,i-1] * (np.pi / 2))
    F_ = F_[:,::-1]
    F_ = r * F_

    np.savetxt(fullpathf, F, delimiter=',', fmt="%1.4e")
    np.savetxt(fullpathx, X, delimiter=',', fmt="%1.4e")
