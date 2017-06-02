import sys
import os
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import paretofront
import vectorutils as vu

def radviz(pf, filepath):
    f_ = pf.get_list('f_')
    m = len(f_[0])
    fs = [math.fsum(f) for f in f_]
    columns = ["f{0:d}".format(i+1) for i in range(m)]
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

    S = [[math.cos(t), math.sin(t)] for t in [2.0 * math.pi * (i/float(m)) for i in range(m)]]
    u1 = [math.fsum([f[j] * c[0] for j,c in enumerate(S)])/fs[i] for i,f in enumerate(f_)]
    u2 = [math.fsum([f[j] * c[1] for j,c in enumerate(S)])/fs[i] for i,f in enumerate(f_)]
    # print(S)
    
    fig = plt.figure()
    ax = plt.gca(xlim = [-1, 1], ylim = [-1, 1])
    ax.scatter(u1, u2, s = ps, marker = 'o', facecolor = rgbs, alpha = 0.4, linewidth = 0.5) 
    ax.add_patch(mpl.patches.Circle((0.0, 0.0), radius = 1.0, facecolor = 'none'))
    for xy, name in zip(S, columns):
        ax.add_patch(mpl.patches.Circle(xy, radius = 0.025, facecolor = 'gray'))
        if xy[0] < 0.0 and xy[1] < 0.0:
            ax.text(xy[0] - 0.025, xy[1] - 0.025, name, ha = 'right', va = 'top', size = 'small')
        elif xy[0] < 0.0 and xy[1] >= 0.0: 
            ax.text(xy[0] - 0.025, xy[1] + 0.025, name, ha = 'right', va = 'bottom', size = 'small')
        elif xy[0] >= 0.0 and xy[1] < 0.0:
            ax.text(xy[0] + 0.025, xy[1] - 0.025, name, ha = 'left', va = 'top', size = 'small')
        elif xy[0] >= 0.0 and xy[1] >= 0.0:
            ax.text(xy[0] + 0.025, xy[1] + 0.025, name, ha = 'left', va = 'bottom', size = 'small')
    ax.axis('equal')

    figfile = filepath.split('-')[0] + "-radvizmu.png"
    print("saving figure:", figfile)
    plt.savefig(figfile, bbox_inches = 'tight')
    print("showing plot")

    plt.show()

if __name__ == "__main__":
    # test()
    
    rootdir = "data"
    # filename = "dtlz63m2000n-mu.csv"
    # filename = "dtlz64m4000n-mu.csv"
    filename = "dtlz65m5000n-mu.csv"
    path = os.path.join(rootdir, filename)

    pf = paretofront.frontdata()
    print("loading file:", path)
    pf.load_csv(path)
    
    radviz(pf, path)
