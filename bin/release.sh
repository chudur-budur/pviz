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

while getopts "bctp" o; do
	case "${o}" in
		c)
			mode=0; # echo "hit -c mode: 0";
			;;
		b)
			mode=1; # echo "hit -b mode: 1";
			;;
        t)  
            mode=2
            ;;
        p)  
            mode=3
            ;;
		*)
			echo "error: some of the parameters are missing, hence exiting ...";
			_usage
			;;
	esac
done
shift $((OPTIND-1))

if [[ $mode == 0 ]]; then
    echo "cleaning build."
    rm -rf build;
    echo "cleaning dist."
    rm -rf dist;
    echo "cleaning viz.egg-info"
    rm -rf viz.egg-info;
elif [[ $mode == 1 ]]; then
    python3 setup.py sdist bdist_wheel
elif [[ $mode == 2 ]]; then
    python3 -m twine upload --repository testviz dist/* --verbose
fi

