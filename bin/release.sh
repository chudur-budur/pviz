#!/bin/bash

shopt -s extglob ;
cpath=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd ) ;

# run it from one folder up, ./sh/release.sh

mode=0;

_usage()
{
	# example: ./maprun -s -i -n 30 -c "ls -al"
	printf '%s\n\n' "Usage: ./sh/release [-cbitp]" ;
	1>&2; exit 1 ;
}

while getopts "cbitp" o; do
	case "${o}" in
		c)
			mode=0; # echo "hit -c mode: 0";
			;;
		b)
			mode=1; # echo "hit -b mode: 1";
			;;
        i)
            mode=2
            ;;
        t)  
            mode=3
            ;;
        p)  
            mode=4
            ;;
		*)
			echo "error: some of the parameters are missing, hence exiting ...";
			_usage
			;;
	esac
done
shift $((OPTIND-1))

if [[ $mode == 0 ]]; then
    # -c, clean
    echo "cleaning build."
    rm -rf build;
    echo "cleaning dist."
    rm -rf dist;
    echo "cleaning pviz.egg-info"
    rm -rf pviz.egg-info;
elif [[ $mode == 1 ]]; then
    # -b, build
    python3 setup.py sdist bdist_wheel
elif [[ $mode == 2 ]]; then
    # -i, install local
    python3 setup.py install
elif [[ $mode == 3 ]]; then
    # -t, publish on pypi-test
    python3 -m twine upload --repository testpypi dist/* --verbose
elif [[ $mode == 4 ]]; then
    # -p, publish on pypi
    python3 -m twine upload --repository pypi dist/* --verbose
fi

