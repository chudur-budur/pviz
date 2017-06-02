import sys
import os
import math
import vectorutils as vu
import paretofront

def compute_mu(pf, epsilon = 0.05):
    """
        Calculate the trade-off weight mu(xi,xj) described in the paper.
        The neighbourhood will be considered as epsilon neighbour, all the
        objective function vector needs to be normalized.
    """
    idx = pf.sorted_indices()
    # idx = [82, 218]
    M = len(pf.data[0].f)
    for i in idx:
        nbrs = pf.nbr_normeps(i, 'y_', epsilon)
        if len(nbrs) > 0:
            # print("nbrs ({0:d}): ".format(i), nbrs)
            w = []
            js = {}
            for j in nbrs:
                gain = 0
                loss = 0
                # i: 82 [0.221, 0.224]
                # j: 218 [0.221, 0.225]
                for m in range(M):
                    gain = gain + max(0, pf.data[j].f_[m] - pf.data[i].f_[m])
                    loss = loss + max(0, pf.data[i].f_[m] - pf.data[j].f_[m])         
                ratio = gain/float(loss) if loss > 0 else float('inf')
                if ratio < float('inf'):
                    w.append(ratio)
                    js[j] = ratio
            # print("w ({0:d}): ".format(i), vu.tostr(w))
            # print("js: ", js)
            if len(w) > 0:
                pf.data[i].mu = min(w)
                pf.data[i].mu_ = min(w)/(max(pf.data[i].f_) - min(pf.data[i].f_))
                # pf.data[i].mu = vu.mean(w)
                # pf.data[i].mu = vu.median(w)
                # print("mu ({0:d}):".format(i), pf.data[i].mu)
            else:
                raise Exception("the w list is empty!! no viable trade-offs in the neighbourhood?")
        else:
            raise Exception("empty neighbourhood! too small epsilon?")

def save_csv(pf, filename):
    """
        Save data into file
    """
    fp = open(filename, 'w')
    fp.write("# {0:s}\n".format(str(pf.header)))
    idx = pf.sorted_indices(lambda i: pf.data[i].mu, reverse = True)
    for i in idx:
        line = str(i) + ',' \
                + vu.tocsvstr(pf.data[i].f) + ',' \
                + vu.tocsvstr(pf.data[i].f_) + ',' \
                + vu.tocsvstr(pf.data[i].x) + ',' \
                + "{0:.3f}".format(pf.data[i].mu) + ',' \
                + "{0:.3f}".format(pf.data[i].mu_) + ',' \
                + vu.tocsvstr(pf.data[i].y) + ',' \
                + vu.tocsvstr(pf.data[i].y_) + '\n'
        fp.write(line)
    fp.close()

if __name__ == "__main__":
    rootdir = "data"

    filelist = [\
    "debmdk2m250n-tsne.csv", "debmdk3m500n-tsne.csv", "debmdk4m1000n-tsne.csv", "debmdk5m2000n-tsne.csv", \
    "dtlz62m500n-tsne.csv", "dtlz63m2000n-tsne.csv", "dtlz64m4000n-tsne.csv", "dtlz65m5000n-tsne.csv", \
    "debiso2m1000n-tsne.csv", "debiso3m3000n-tsne.csv", "debiso4m4000n-tsne.csv", "debiso5m5000n-tsne.csv", \
    "slantgauss2m500n-tsne.csv", "slantgauss3m1000n-tsne.csv", "slantgauss4m2000n-tsne.csv"\
    ]
    
    # filename = "debmdk2m250n-tsne.csv"
    # filename = "debmdk3m500n-tsne.csv"
    # filename = "debmdk4m1000n-tsne.csv"
    # filename = "debmdk5m2000n-tsne.csv"
    
    # filename = "dtlz63m2000n-tsne.csv"
    # filename = "dtlz64m4000n-tsne.csv"
    # filename = "dtlz65m5000n-tsne.csv"
    
    # filename = "debiso2m1000n-tsne.csv"
    # filename = "debiso3m3000n-tsne.csv"
    # filename = "debiso4m4000n-tsne.csv"
    # filename = "debiso5m5000n-tsne.csv"
    
    # filename = "slantgauss2m500n-tsne.csv"
    # filename = "slantgauss3m1000n-tsne.csv"
    # filename = "slantgauss4m2000n-tsne.csv"
    
    for filename in filelist:
        filepath = os.path.join(rootdir, filename)

        pf = paretofront.frontdata()

        print("loading file:", filepath)
        pf.load_csv(filepath)

        print("re-computing mu from normalized Y")
        compute_mu(pf, 0.150)

        outfile = filepath.split('-')[0] + "-wsntsne.csv"
        print("saving file:", outfile)
        save_csv(pf, outfile)
