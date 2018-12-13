import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utils

# prefix = "data/spherical/spherical-3d-norm"
# prefix = "data/spherical/spherical-4d-norm"
# prefix = "data/spherical/spherical-8d-norm"

# prefix = "data/spherical-equidist/spherical-equidist-3d-norm"

# prefix = "data/knee/knee-3d-norm"
# prefix = "data/knee/knee-4d-norm"
# prefix = "data/knee/knee-8d-norm"

# prefix = "data/isolated/isolated-3d-norm"
# prefix = "data/isolated/isolated-4d-norm"
# prefix = "data/isolated/isolated-8d-norm"

# prefix = "data/line/line-3d-norm"
# prefix = "data/line/line-4d-norm"
# prefix = "data/line/line-6d-norm"
# prefix = "data/line/line-8d-norm"

# prefix = "data/carcrash/carcrash-3d-norm"
# prefix = "data/carcrash-c1/carcrash-c1-3d-norm"
# prefix = "data/carcrash-c2/carcrash-c2-3d-norm"
# prefix = "data/carcrash/carcrash-c1/carcrash-c1-3d-norm"
# prefix = "data/carcrash/carcrash-c2/carcrash-c2-3d-norm"

# Do all these plots with constraint based coloring

# prefix = "data/knee-const/knee-const-3d-norm"
# prefix = "data/knee-const/knee-const-4d-norm"
# prefix = "data/knee-const/knee-const-8d-norm"

# prefix = "data/c2dtlz2/c2dtlz2-3d-norm"
# prefix = "data/c2dtlz2/c2dtlz2-4d-norm"
prefix = "data/c2dtlz2/c2dtlz2-5d-norm"
# prefix = "data/c2dtlz2/c2dtlz2-8d-norm"

# prefix = "data/c2dtlz2/c2dtlz2-c1/c2dtlz2-c1-3d-norm"
# prefix = "data/c2dtlz2/c2dtlz2-c2/c2dtlz2-c2-3d-norm"
# prefix = "data/c2dtlz2/c2dtlz2-c3/c2dtlz2-c3-3d-norm"
# prefix = "data/c2dtlz2/c2dtlz2-c4/c2dtlz2-c4-3d-norm"

# prefix = "data/gaa-das/gaa-das-10d-norm"
# prefix = "data/gaa-lhs/gaa-lhs-10d-norm"
# prefix = "data/gaa-lhs/gaa-lhs-c1/gaa-lhs-c1-10d-norm"
# prefix = "data/gaa-lhs/gaa-lhs-c2/gaa-lhs-c2-10d-norm"
# prefix = "data/gaa-lhs/gaa-lhs-c3/gaa-lhs-c3-10d-norm"

sns.set()
points = utils.load(prefix + ".out")
sorted(points)
data = np.array(points)
plt.figure()
ax = sns.heatmap(data)
outfile_name = prefix + "-heatmap.pdf"
plt.savefig(outfile_name, transparent = False, bbox_inches = 'tight', pad_inches = 0)
plt.show()
