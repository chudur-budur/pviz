import math
import vectorutils as vu
import utils


if __name__ == "__main__":
    data_file = "data/iris/iris-4d.out"
    points = utils.load(data_file)
    [lb, ub] = vu.get_bound(points)
    print(lb, ub)

