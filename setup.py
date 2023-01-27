from setuptools import find_packages, setup

setup(
    name='pySODM',
    packages=find_packages("src", exclude=["*.tests"]),
    package_dir={'': 'src'},
    version='0.2.1',
    description='Simulating and Optimising Dynamical Models',
    author='Tijs Alleman, KERMIT, Ghent University',
    author_email='tijs.alleman@ugent.be',
    keywords='ODE PDE simulation calibration gillespie xarray emcee',
    url='https://github.com/twallema/pySODM',
    license='MIT',
    install_requires=[
        'scipy',
        'numpy',
        'pandas',
        'numba',
        'matplotlib',
        'xarray',
        'emcee',
        'h5py'
    ],
    extras_require={
        "develop":  ["pytest",
                     "sphinx",
                     "numpydoc",
                     "sphinx_rtd_theme",
                     "myst_parser[sphinx]"],
    }
)
