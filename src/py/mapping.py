import math
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
import utils
import vectorutils as vu

def mapradviz(data, theta):
    fs = [math.fsum(v) for v in data]
    S = [[math.cos(rad), math.sin(rad)] for rad in [(t * math.pi) / 180.0 for t in theta]]
    u = [math.fsum([f[j] * s[0] for j,s in enumerate(S)])/fs[i] for i,f in enumerate(data)]
    v = [math.fsum([f[j] * s[1] for j,s in enumerate(S)])/fs[i] for i,f in enumerate(data)]
    coords = list(zip(u, v))
    return [list(v) for v in coords]

def error(x, points, nbr):
    epsilon = 0.25
    x_ = [(v * 359.0) for v in x]
    rvpoints_ = mapradviz(points, x_)
    [lb, ub] = vu.get_bound(rvpoints_)
    rvpoints = vu.normalize(rvpoints_, lb, ub)
    # rvnbr = NearestNeighbors()
    # rvnbr.fit(rvpoints)
    err = 0.0
    count = 0
    for i,point in enumerate(points):
        nbrs = nbr.radius_neighbors([point], epsilon, \
                return_distance = False)[0].tolist()
        for j in nbrs:
            if vu.distlp(list(rvpoints[i]), list(rvpoints[j])) > epsilon:
                count = count + 1
    return (count / len(rvpoints))

if __name__ == "__main__":
    points = utils.load("data/airofoil/airofoil-6d-norm.out")
    
    nbr = NearestNeighbors()
    nbr.fit(points)

    theta = [0.0, 90, 180, 270]
    rvpoints = mapradviz(points, theta)
    print(len(points), points[0:10])
    print(len(rvpoints), rvpoints[0:10])

    x0 = [0.0, 0.25, 0.5, 0.75]
    print(error(x0, points, nbr))

    x1 = [0.125, 0.135, 0.145, 0.155]
    print(error(x1, points, nbr))

    res = []
    for a in range(1, 7):
        for b in range(1, 7):
            for c in range(1, 7):
                for d in range(1, 7):
                    x = [a * 0.15, b * 0.15, c * 0.15, d * 0.15]
                    print(x, "-->", error(x, points, nbr))
                    res.append([error, x])
    res.sort()
    print(res)

    fp = open("mapping.out", "w")
    for t in res:
        fp.write("{0:f}".format(t[0]) + "\t" \
                + "\t".join(["{0:f}".format(v) for v in t]) + "\n")
    fp.close()
    
    """
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    res = minimize(error, x1, args = (points, nbr), method='L-BFGS-B', \
            bounds = bounds, options = {'disp': True})
    print("res:", res.x)
    """
