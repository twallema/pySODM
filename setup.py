from setuptools import find_packages, setup

setup(
    name='pySODM',
    packages=find_packages("src", exclude=["*.tests"]),
    package_dir={'': 'src'},
    version='0.1.1',
    description='Simulating and Optimising Dynamical Models',
    author='Tijs Alleman, KERMIT, Ghent University',
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
