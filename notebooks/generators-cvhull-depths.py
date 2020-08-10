"""generators-cvhull-depths.py -- a python script from generators-cvhull-depths.ipynb to be run on hpcc.

    Just a copy of `generators-cvhull-depths.ipynb` as a python script.
"""

import sys
import os
import numpy as np
sys.path.append('../')
from vis.tda import simple_shape
from vis.utils import io

pfs = {'dtlz2': ['3d', '4d', '8d'], \
       'dtlz2-nbi': ['3d', '4d', '8d'], \
       'debmdk': ['3d', '4d', '8d'], \
       'debmdk-nbi': ['3d', '4d', '8d'], \
       'debmdk-all': ['3d', '4d', '8d'], \
       'debmdk-all-nbi': ['3d', '4d', '8d'], \
       'dtlz8': ['3d', '4d', '6d', '8d'], \
       'dtlz8-nbi': ['3d', '4d', '6d', '8d'], \
       'c2dtlz2': ['3d', '4d', '5d', '8d'], \
       'c2dtlz2-nbi': ['3d', '4d', '5d', '8d'], \
       'cdebmdk': ['3d', '4d', '8d'], \
       'cdebmdk-nbi': ['3d', '4d', '8d'], \
       'c0dtlz2': ['3d', '4d', '8d'], \
       'c0dtlz2-nbi': ['3d', '4d', '8d'], \
       'crash-nbi': ['3d'], 'crash-c1-nbi': ['3d'], 'crash-c2-nbi': ['3d'], \
       'gaa': ['10d'], \
       'gaa-nbi': ['10d']}

for pf in list(pfs.keys())[-2:]:
    for dim in pfs[pf]:
        fullpathf = "../data/{0:s}/{1:s}/dataf.csv".format(pf, dim)
        if os.path.exists(fullpathf):
            path, filenamef = os.path.split(fullpathf)
            dirs = path.split('/')
            frontname = dirs[-2]

            F = np.loadtxt(fullpathf, delimiter=',')
            print(fullpathf, F.shape, dirs, frontname)
            
            # test simple_shape.depth_contour function
            # it looks like these PFs are better displayed if project_collapse=False
            if pf in ['dtlz8', 'dtlz8-nbi', 'crash-nbi', 'crash-c1-nbi', 'crash-c2-nbi']:
                L = simple_shape.depth_contours(F, project_collapse=False)
            elif pf in ['gaa', 'gaa-nbi']:
                L = simple_shape.depth_contours(F, verbose=True)
            else:
                L = simple_shape.depth_contours(F)
            # save the layers
            io.savetxt(os.path.join(path, "depth-cont-cvhull.csv"), L, fmt='{:d}', delimiter=',')
            
            # We are not using this since it's exrtemely slow and also doesn't give
            # layers if all the points are on a fully convex surface.
            # print("Generating depth-contours (project_collapse=False) for " + frontname)
            # # test ss.depth_contour function without projection and collapse
            # L = ss.depth_contours(F, project_collapse = False)
            # save the layers
            # io.savetxt(os.path.join(path, "depth-cont-cvhull.csv"), L, fmt = '{:d}', delimiter = ',')
