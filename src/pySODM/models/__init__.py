"""
By setting the following environment variables, pySODM prohibits Numpy, Torch & Tensorflow from automatically multithreading large matrix operations
This does not play nice with the pySODM feature of running the model N times in parallel using multiprocessing
"""

import os

# Numpy & Torch
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Tensorflow
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'