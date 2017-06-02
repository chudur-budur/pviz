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
        nbrs = pf.nbr_normeps(i, 'f_', epsilon)
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

def compute_mu_knn(pf, k = 3):
    """
        Calculate the trade-off weight mu(xi,xj) described in the paper.
        The neighbourhood will be considered as knn neighbour, all the
        objective function vector needs to be normalized.
    """
    idx = pf.sorted_indices()
    M = len(pf.data[0].f)
    for i in idx:
        nbrs = pf.nbr_knn(i, 'f_', k)
        if len(nbrs) > 0:
            print("nbrs ({0:d}): ".format(i), nbrs)
            w = []
            js = {}
            for j in nbrs:
                gain = 0
                loss = 0
                for m in range(M):
                    gain = gain + max(0, pf.data[j].f_[m] - pf.data[i].f_[m])
                    loss = loss + max(0, pf.data[i].f_[m] - pf.data[j].f_[m])         
                ratio = gain/float(loss) if loss > 0 else float('inf')
                if ratio < float('inf'):
                    w.append(ratio)
                    js[j] = ratio
            if len(w) > 0:
                pf.data[i].mu = min(w)
                pf.data[i].mu_ = min(w)/(max(pf.data[i].f_) - min(pf.data[i].f_))
                # pf.data[i].mu = vu.mean(w)
                # pf.data[i].mu = vu.median(w)
                print("mu ({0:d}):".format(i), pf.data[i].mu)
            else:
                raise Exception("the w list is empty!! no viable trade-offs in the neighbourhood?")
        else:
            raise Exception("empty neighbourhood! too small epsilon?")

def save_csv(pf, filename):
    """
        Save data into file
    """
    fp = open(filename, 'w')
    s1 = pf.header['x'][1] + 1
    e1 = s1
    pf.header['mu'] = [s1, e1]
    s1 = s1 + 1
    e1 = s1
    pf.header['mu_'] = [s1, e1]
    fp.write("# {0:s}\n".format(str(pf.header)))
    idx = pf.sorted_indices(lambda i: pf.data[i].mu, reverse = True)
    for i in idx:
        line = str(i) + ',' \
                + vu.tocsvstr(pf.data[i].f) + ',' \
                + vu.tocsvstr(pf.data[i].f_) + ',' \
                + vu.tocsvstr(pf.data[i].x) + ',' \
                + "{0:.3f}".format(pf.data[i].mu) + ',' \
                + "{0:.3f}".format(pf.data[i].mu_) + '\n'
        fp.write(line)
    fp.close()

if __name__ == "__main__": 
    """
        The main function
    """

    rootdir = "data"
    filelist = [\
            "debmdk2m250n.csv", "debmdk3m500n.csv", "debmdk4m1000n.csv", "debmdk5m2000n.csv", \
            "dtlz62m500n.csv", "dtlz63m2000n.csv", "dtlz64m4000n.csv", "dtlz65m5000n.csv", \
            "debiso2m1000n.csv", "debiso3m3000n.csv", "debiso4m4000n.csv", "debiso5m5000n.csv", \
            "slantgauss2m500n.csv", "slantgauss3m1000n.csv", "slantgauss4m2000n.csv" \
            ]
    
    for filename in filelist:
        infile = os.path.join(rootdir, filename)
        pf = paretofront.frontdata()

        print("loading file:", infile)
        pf.load_csv(infile)
        
        M = pf.header['M']
        eps = {2:0.125, 3:0.250, 4:0.500, 5:0.5}
        print("computing mu values")
        # 2d: eps = 0.125, i = 4th,  k = 5, i = ?
        # 3d: eps = 0.250, i = 1st,  k = 8, i = ?
        # 4d: eps = 0.500, i = 7th,  k = ?, i = ?
        # 5d: eps = 0.500, i = 14th, k = ?, i = ? 
        # 6d: eps = 0.500, i = 29th, k = ?, i = ?
        # compute_mu_knn(pf, 8)
        # compute_mu(pf, eps[M])
        compute_mu(pf, 0.5)

        outfile = infile.split('.')[0] + "-mu.csv"
        print("saving file:", outfile)
        save_csv(pf, outfile)
