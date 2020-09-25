#!/bin/bash

shopt -s extglob ;
cpath=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd ) ;

# run it from one folder up, ./sh/release.sh

mode=0;

_usage()
{
	# example: ./maprun -s -i -n 30 -c "ls -al"
	printf '%s\n\n' "Usage: ./sh/release [-bc]" ;
	1>&2; exit 1 ;
}

while getopts "bc" o; do
	case "${o}" in
		b)
			mode=1; # echo "hit -b mode: 1";
			;;
		c)
			mode=0; # echo "hit -c mode: 0";
			;;
		*)
			echo "error: some of the parameters are missing, hence exiting ...";
			_usage
			;;
	esac
done
shift $((OPTIND-1))

if [[ $mode == 1 ]]; then
    pipenv run python3 setup.py sdist bdist_wheel
else
    echo "cleaning build."
    rm -rf build;
    echo "cleaning dist."
    rm -rf dist;
    echo "cleaning viz.egg-info"
    rm -rf viz.egg-info;
fi

