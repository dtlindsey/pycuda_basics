import numpy as np

# -- PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

#####################
# iDivUp FUNCTION
####################
def iDuvUp(a, b):
    return a // b + 1

NUM_THREADS = 100000
BLOCK_SIZE = 256

def match(val_check):
    """
    Args:
        val_check: numpy float32 array of len 3

    Returns: bool

    """
    return_val = np.bool
    # -- Allocate GPU device memory
    data_a = cuda.mem_alloc(val_check.nbytes)
    cuda.memcpy_htod(data_a, val_check)


    mod = SourceModule("""
    #include <stdio.h>
    __global__ void deviceCheckValid(float * val_check)
    """)