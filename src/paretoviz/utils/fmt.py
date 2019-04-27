"""
This file contains different system related utility functions.
"""

def cat(vals, dtype = 'float'):
    """
    Just cats a list of lists.
    """
    if dtype == 'float':
        fmt = "{:.4e}"
    elif dtype == 'int':
        fmt = "{:d}"
    elif dtype == 'str':
        fmt = "{:s}"
    for v in vals:
        if type(v) is list:
            print("\t".join([fmt.format(x) for x in v]))
        else:
            print(fmt.format(v))

def save(vals, path, sep = '\t', dtype = 'float'):
    """
    Saves data into file.
    """
    fp = open(path, 'w')
    if dtype == 'float':
        fmt = "{:.4e}"
    elif dtype == 'int':
        fmt = "{:d}"
    elif dtype == 'str':
        fmt = "{:s}"
    for v in vals:
        if type(v) is list:
            fp.write(sep.join([fmt.format(x) for x in v]) + "\n")
        else:
            fp.write(fmt.format(v) + "\n")
    fp.close()

def load(path, sep = None, dtype = 'float'):
    """
    Load data from file.
    """
    fp = open(path, 'r')
    data = []
    if dtype == 'float':
        for line in fp:
            vals = line.strip().split(sep)
            if len(vals) > 1:
                data.append([float(v) for v in vals])
            else:
                data.append([float(vals[0])])
    elif dtype == 'int':
        for line in fp:
            vals = line.strip().split(sep)
            if len(vals) > 1:
                data.append([int(v) for v in vals])
            else:
                data.append([int(vals[0])])
    elif dtype == 'str':
        for line in fp:
            vals = line.strip().split(sep)
            if len(vals) > 1:
                data.append(vals)
            else:
                data.append([vals[0]])
    fp.close()
    return data

if __name__ == "__main__":
    print("__main__ has nothing as of now.")
    pass
