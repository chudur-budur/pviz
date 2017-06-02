import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import paretofront
import vectorutils as vu

def plot_tsne_mu(pf, filepath):
    """
        Displays the t-SNE scatter plot from the saved csv data file.
    """
    muvals = np.array(pf.get_list('mu'))
    mumax = max(muvals)
    mumin = min(muvals)
    munorm = vu.normalize(muvals, mumin, mumax)
    Y = np.array(pf.get_list('y_'))
    
    normc = mpl.colors.Normalize(vmin = 0.0, vmax = 1.0)
    # cmap = mpl.cm.Greys
    # cmap = mpl.cm.RdBu_r
    cmap = mpl.cm.jet
    clrmap = mpl.cm.ScalarMappable(norm = normc, cmap = cmap)
    rgbs = [clrmap.to_rgba(v) for v in munorm]
    ps = [int(10 + (v * 100)) for v in munorm]

    fig = plt.figure()
    fig.canvas.set_window_title(filepath)
    plt.scatter(Y[:, 0], Y[:, 1], s = ps, marker = 'o', facecolor = rgbs, alpha = 0.4, linewidth = 0.5)
    
    figfile = filepath.split('.')[0] + "mu.png"
    plt.savefig(figfile, bbox_inches = 'tight')
    
    plt.show()

def plot_tsne_munorm(pf, filepath):
    """
        Displays the t-SNE scatter plot from the saved csv data file.
    """
    muvals = np.array(pf.get_list('mu_'))
    mumax = max(muvals)
    mumin = min(muvals)
    munorm = vu.normalize(muvals, mumin, mumax)
    Y = np.array(pf.get_list('y_'))
    
    normc = mpl.colors.Normalize(vmin = 0.0, vmax = 1.0)
    # cmap = mpl.cm.Greys
    # cmap = mpl.cm.RdBu_r
    cmap = mpl.cm.jet
    clrmap = mpl.cm.ScalarMappable(norm = normc, cmap = cmap)
    rgbs = [clrmap.to_rgba(v) for v in munorm]
    ps = [int(10 + (v * 100)) for v in munorm]

    fig = plt.figure()
    fig.canvas.set_window_title(filepath)
    plt.scatter(Y[:, 0], Y[:, 1], s = ps, marker = 'o', facecolor = rgbs, alpha = 0.4, linewidth = 0.5)
    
    figfile = filepath.split('.')[0] + "munrom.png"
    plt.savefig(figfile, bbox_inches = 'tight')
    
    plt.show()

def plot_knees_mu(pf, filepath, axes = [0, 1, 2]):
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
    
    figfile = filepath.split('-')[0] + "-mupf.png"
    print("saving figure:", figfile)
    plt.savefig(figfile, bbox_inches = 'tight')
    print("showing plot")
    plt.show()

def plot_knees_munorm(pf, filepath, axes = [0, 1, 2]):
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
    
    figfile = filepath.split('-')[0] + "-munormpf.png"
    print("saving figure:", figfile)
    plt.savefig(figfile, bbox_inches = 'tight')
    print("showing plot")
    plt.show()

if __name__ == "__main__":
    """
        Main code
    """
    rootdir = "data"

    filelist = [\
    "debmdk2m250n-tsne.csv", "debmdk3m500n-tsne.csv", "debmdk4m1000n-tsne.csv", "debmdk5m2000n-tsne.csv", \
    "dtlz62m500n-tsne.csv", "dtlz63m2000n-tsne.csv", "dtlz64m4000n-tsne.csv", "dtlz65m5000n-tsne.csv", \
    "debiso2m1000n-tsne.csv", "debiso3m3000n-tsne.csv", "debiso4m4000n-tsne.csv", "debiso5m5000n-tsne.csv", \
    "slantgauss2m500n-tsne.csv", "slantgauss3m1000n-tsne.csv", "slantgauss4m2000n-tsne.csv"\
    ] 
    # for filename in filelist:
    #     filepath = os.path.join(rootdir, filename)
    #     pf = paretofront.frontdata()
   
    #     print("loading file:", filename)
    #     pf.load_csv(filepath)
    #     print("showing plot")
    #     plot_tsne_mu(pf, filepath)
    #     plot_tsne_munorm(pf, filepath)
    
    # mu from tsne neighbourhood
    # filename = "debmdk2m250n-wsntsne.csv"
    # filename = "debmdk3m500n-wsntsne.csv"
    # filename = "debmdk4m1000n-wsntsne.csv"
    # filename = "debmdk5m2000n-wsntsne.csv"
    
    # filename = "dtlz63m2000n-wsntsne.csv"
    # filename = "dtlz64m4000n-wsntsne.csv"
    # filename = "dtlz65m5000n-wsntsne.csv"
    
    # filename = "debiso2m1000n-wsntsne.csv"
    # filename = "debiso3m3000n-wsntsne.csv"
    # filename = "debiso4m4000n-wsntsne.csv"
    # filename = "debiso5m5000n-wsntsne.csv"
    
    # filename = "slantgauss2m500n-wsntsne.csv"
    # filename = "slantgauss3m1000n-wsntsne.csv"
    # filename = "slantgauss4m2000n-wsntsne.csv"

    filelistwsn = [\
    "debmdk2m250n-wsntsne.csv", "debmdk3m500n-wsntsne.csv", \
    "debmdk4m1000n-wsntsne.csv", "debmdk5m2000n-wsntsne.csv", \
    "dtlz62m500n-wsntsne.csv", "dtlz63m2000n-wsntsne.csv", \
    "dtlz64m4000n-wsntsne.csv", "dtlz65m5000n-wsntsne.csv", \
    "debiso2m1000n-wsntsne.csv", "debiso3m3000n-wsntsne.csv", \
    "debiso4m4000n-wsntsne.csv", "debiso5m5000n-wsntsne.csv", \
    "slantgauss2m500n-wsntsne.csv", "slantgauss3m1000n-wsntsne.csv", \
    "slantgauss4m2000n-wsntsne.csv"\
    ]
    for filename in filelistwsn:
        filepath = os.path.join(rootdir, filename)
        pf = paretofront.frontdata()
   
        print("loading file:", filename)
        pf.load_csv(filepath)
        print("showing plot")
        plot_tsne_mu(pf, filepath)
        plot_tsne_munorm(pf, filepath)
        if pf.header['M'] <= 3:
            plot_knees_mu(pf, filepath)
            plot_knees_munorm(pf, filepath)
