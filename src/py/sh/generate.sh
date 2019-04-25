#!/bin/bash
#comments: This script generates all data points for the visualization experiment. 
# for the line-surface pf, use the matlab code.

shopt -s extglob
cpath=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

names=(spherical knee knee-const isolated c2dtlz2)
# names=(c2dtlz2)
mode=$1

if [[ -z $mode ]]; then
    mode="uniform"
fi

for name in ${names[*]}
do
    if [ "$name" = "isolated" ]; then
        mode="random"
    fi
    echo "Generating surface for 3d $name"
    python3 ${name}.py 3 $mode
    echo "Generating surface for 4d $name"
    python3 ${name}.py 4 $mode
    if [ "$name" = "c2dtlz2" ]; then
        echo "Generating surface for 5d $name"
        python3 ${name}.py 5 $mode
    fi
    echo "Generating surface for 8d $name"
    python3 ${name}.py 8 $mode
done
