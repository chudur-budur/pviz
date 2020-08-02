#!/bin/bash

srcroot="/tmp/data"
destroot="/Users/khaled/data/research/visualization/data"
fname="depth-cont-cvhull.csv"

pfs=(c0dtlz2 c0dtlz2-nbi c2dtlz2 c2dtlz2-nbi cdebmdk cdebmdk-nbi crash-c1-nbi crash-c2-nbi crash-nbi debmdk debmdk-all debmdk-all-nbi debmdk-nbi dtlz2 dtlz2-nbi dtlz8 dtlz8-nbi gaa gaa-nbi)
dims=(3d 4d 5d 6d 7d 8d 10d)

for pf in ${pfs[*]}
do
    for dim in ${dims[*]}
    do
        srcpath="$srcroot/$pf/$dim/$fname"
        destpath="$destroot/$pf/$dim/$fname"
        if [ -f "$srcroot/$pf/$dim/$fname" ]; then 
            echo "$srcpath"
            if [ -d "$destroot/$pf/$dim" ]; then
                echo "$destpath"
                cp $srcpath $destpath
            fi
        fi
    done
done
