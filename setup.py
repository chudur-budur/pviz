import setuptools
from setuptools import setup
import os

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
    print("long_description path = " + f.name)
    long_description = f.read()

version = {}
with open(os.path.join(_here, 'viz', 'version.py')) as f:
    print("version path = " + f.name)
    exec(f.read(), version)

pack = setuptools.find_packages()
print("packages =", pack)

setup(
    name='pviz',
    version=version['__version__'],
    author='AKM Khaled Talukder',
    author_email='talukde1@msu.edu',
    description=(\
            'A framework for high-dimensional (3 <= m <= 10) Pareto-optimal ' \
            + 'front visualization and analytics'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/chudur-budur/pviz',
    license='Apache-2.0',
    # packages=['viz', 'viz.utils', 'viz.generators', 'viz.tda', 'viz.plotting'],
    packages=setuptools.find_packages(),
    setup_requires=['scipy==1.5.2', 'matplotlib'],
    install_requires=['scipy==1.5.2', 'matplotlib'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8'
        ],
    )
