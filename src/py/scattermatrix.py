import os
import sys
import math
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import paretofront
import vectorutils as vu

def scatterplot_matrix(pf, filepath):
    # get all the plot data and specs
    f_ = pf.get_list('f_')
    M = pf.header['M']
    n = pf.header['n']
    names = ["f{0:d}".format(i+1) for i in range(M)]
    data = [[f[i] for f in f_] for i in range(M)]
    mu = pf.get_list('mu_')
    mumin = min(mu)
    mumax = max(mu)
    munorm = vu.normalize(mu, mumin, mumax)

    normc = mpl.colors.Normalize(vmin = 0.0, vmax = 1.0)
    # cmap = mpl.cm.Greys
    # cmap = mpl.cm.RdBu_r
    cmap = mpl.cm.jet
    clrmap = mpl.cm.ScalarMappable(norm = normc, cmap = cmap)
    rgbs = [clrmap.to_rgba(v) for v in munorm]
    ps = [int(10 + (v * 100)) for v in munorm]
    
    # make the subplot matrix
    fig, axes = plt.subplots(nrows = M, ncols = M, figsize = (8,8))
    fig.subplots_adjust(hspace = 0.05, wspace = 0.05)
    
    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')
    
    # only the upper and the lower triangular part will have the plot
    for i,j in [(r,c) for c in range(M) for r in range(c)]:
        for x,y in [(i,j), (j,i)]:
            axes[x,y].scatter(data[x], data[y], \
                s = ps, marker = 'o', facecolor = rgbs, alpha = 0.4, linewidth = 0.5)
    
    # put the names in the diagonal subplots
    for i,label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords = 'axes fraction', \
                ha = 'center', va = 'center')
    
    # now turn on the ticks
    # for i, j in zip(range(M), itertools.cycle((-1, 0))):
    #     axes[j,i].xaxis.set_visible(True)
    #     axes[i,j].yaxis.set_visible(True)
    
    figfile = filepath.split('-')[0] + "-spmatmu.png"
    print("saving figure:", figfile)
    plt.savefig(figfile, bbox_inches = 'tight')
    print("showing plot")

    plt.show()


if __name__ == "__main__":
    rootdir = "data"
    # filename = "dtlz62m500n-mu.csv"
    # filename = "dtlz63m2000n-mu.csv"
    # filename = "dtlz64m4000n-mu.csv"
    # filename = "dtlz65m5000n-mu.csv"
    
    # filename = "debiso2m1000n-mu.csv"
    # filename = "debiso3m3000n-mu.csv"
    # filename = "debiso4m4000n-mu.csv"
    # filename = "debiso5m5000n-mu.csv"
    
    filename = "slantgauss4m2000n-mu.csv"
    
    path = os.path.join(rootdir, filename)

    pf = paretofront.frontdata()
    print("loading file:", path)
    pf.load_csv(path)

    scatterplot_matrix(pf, path)
