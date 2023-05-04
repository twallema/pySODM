from setuptools import find_packages, setup

setup(
    name='pySODM',
    packages=find_packages("src", exclude=["*.tests"]),
    package_dir={'': 'src'},
    version='0.2.3',
    description='Simulating and Optimising Dynamical Models',
    author='Tijs Alleman, KERMIT, Ghent University',
    author_email='tijs.alleman@ugent.be',
    keywords='ODE PDE simulation calibration gillespie xarray emcee',
    url='https://github.com/twallema/pySODM',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
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