import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import paretofront
import vectorutils as vu

def plot_data_mu(pf, filepath, axes = [0, 1, 2], attr = None):
    """
        Now plot the data points colored according to the mu() value
        knee points will have darker shade
    """
    muvals = pf.get_list('mu')
    minmu = min(muvals)
    maxmu = max(muvals)
    normmu = vu.normalize(muvals, minmu, maxmu)
    
    x = []
    y = []
    z = []
    for i in range(len(pf.data)):
        x.append(pf.data[i].f[axes[0]])
        y.append(pf.data[i].f[axes[1]])
        if len(pf.data[i].f) > 2:
            z.append(pf.data[i].f[axes[2]])

    normc = mpl.colors.Normalize(vmin = 0, vmax = 1.0)
    # cmap = mpl.cm.Greys
    # cmap = cm.RdBu
    cmap = mpl.cm.jet
    # cmap = cm.rainbow_r
    clrmap = mpl.cm.ScalarMappable(norm = normc, cmap = cmap)
    maxclr = max(muvals)
    rgbs = [clrmap.to_rgba(v) for v in normmu]
    ps = [int(10 + (v * 100)) for v in normmu]
    
    fig = plt.figure()
    fig.canvas.set_window_title(filepath)
    if len(z) > 0:
        ax = Axes3D(fig)
        ax.scatter(x, y, z, s = ps, marker = 'o', facecolor = rgbs, alpha = 0.4, linewidth = 0.5)
    else:
        plt.scatter(x, y, s = ps, marker = 'o', facecolor = rgbs, alpha = 0.4, linewidth = 0.5)
    
    figfile = filepath.split('.')[0] + ".png"
    print("saving figure:", figfile)
    plt.savefig(figfile, bbox_inches = 'tight')
    print("showing plot")
    plt.show()

def plot_data_munorm(pf, filepath, axes = [0, 1, 2]):
    """
        Now plot the data points colored according to the mu() value
        knee points will have darker shade
    """
    muvals = pf.get_list('mu_')
    minmu = min(muvals)
    maxmu = max(muvals)
    normmu = vu.normalize(muvals, minmu, maxmu)
    
    x = []
    y = []
    z = []
    for i in range(len(pf.data)):
        x.append(pf.data[i].f[axes[0]])
        y.append(pf.data[i].f[axes[1]])
        if len(pf.data[i].f) > 2:
            z.append(pf.data[i].f[axes[2]])

    normc = mpl.colors.Normalize(vmin = 0, vmax = 1.0)
    # cmap = mpl.cm.Greys
    # cmap = cm.RdBu
    cmap = mpl.cm.jet
    # cmap = cm.rainbow_r
    clrmap = mpl.cm.ScalarMappable(norm = normc, cmap = cmap)
    maxclr = max(muvals)
    rgbs = [clrmap.to_rgba(v) for v in normmu]
    ps = [int(10 + (v * 100)) for v in normmu]
    
    fig = plt.figure()
    fig.canvas.set_window_title(filepath)
    if len(z) > 0:
        ax = Axes3D(fig)
        ax.scatter(x, y, z, s = ps, marker = 'o', facecolor = rgbs, alpha = 0.4, linewidth = 0.5)
    else:
        plt.scatter(x, y, s = ps, marker = 'o', facecolor = rgbs, alpha = 0.4, linewidth = 0.5)
    
    figfile = filepath.split('.')[0] + "norm.png"
    print("saving figure:", figfile)
    plt.savefig(figfile, bbox_inches = 'tight')
    print("showing plot")
    plt.show()

if __name__ == "__main__":
    rootdir = "data"
    
    filelist = [\
            "debmdk2m250n-mu.csv", "debmdk3m500n-mu.csv", \
            "dtlz62m500n-mu.csv", "dtlz63m2000n-mu.csv", \
            "debiso2m1000n-mu.csv", "debiso3m3000n-mu.csv", \
            "slantgauss2m500n-mu.csv", "slantgauss3m1000n-mu.csv" \
            ]

    for filename in [filelist[5]]:
        filepath = os.path.join(rootdir, filename)

        pf = paretofront.frontdata()

        print("loading file:", filename)
        pf.load_csv(filepath)
        print(pf)
        plot_data_mu(pf, filepath)
        plot_data_munorm(pf, filepath)
    
