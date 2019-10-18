
fp = open("sphere-uniform-8d.out", 'r')

fvals = []
cv = []
for line in fp:
    f = [float(v.strip()) for v in line.strip().split()]
    c = (0.98 - f[-1]) * (f[-1] - 0.75)
    if c <= 0:
        cv.append(c)
        fvals.append(f)
fp.close()

print(fvals[0])

fp = open("isolated-uniform-8d.out", 'w')
for f in fvals:
    fp.write("\t".join(["{0:.4f}".format(v) for v in f]) + "\n")
fp.close()

fp = open("isolated-uniform-8d-cv.out", 'w')
for c in cv:
    fp.write("{0:.4f}\n".format(c))
fp.close()
