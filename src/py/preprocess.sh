#!/bin/bash
# Usage:
#   1. Invoke this script like ./preprocess knee 0.25
#   2. The default value for epsilon is 0.125 and it should work for most cases.
#   3. For knee problem set the epsilon to 0.25. But for the constrained knee problem, 
#       set epsilon to default value.

shopt -s extglob
cpath=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

fname=$1
epsilon=$2

if [[ -z $epsilon ]]; then
	epsilon="0.125" ;
    if [ "$fname" = "knee" ]; then
        epsilon="0.25"
    fi
    if [ "$fname" = "line" ]; then
        epsilon="0.025"
    fi
fi

dims=(3 4 5 6 8)

for dim in ${dims[*]}
do
    if [ -f "data/$fname/$fname-${dim}d.out" ]; then
        echo "Processing $fname-${dim}d with epsilon of $epsilon ..."
        python3 normalize.py    data/$fname/$fname-${dim}d.out
        python3 tradeoff.py     data/$fname/$fname-${dim}d-norm.out $epsilon
        python3 peel.py         data/$fname/$fname-${dim}d-norm.out
        if [ "$fname" = "isolated" ]; then
            python3 palettize.py    data/$fname/$fname-${dim}d-norm.out 3
        else
            python3 palettize.py    data/$fname/$fname-${dim}d-norm.out 4
        fi
    fi
done
