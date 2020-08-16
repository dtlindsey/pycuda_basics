import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import datetime

import numpy as np
a = np.random.randn(4, 4)
a = a.astype(np.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
__global__ void doublify(float *a)
{
int idx = threadIdx.x + threadIdx.y*4;
a[idx] *= 2;
}
""")

func = mod.get_function("doublify")
func(a_gpu, block=(4, 4, 1))

start = datetime.datetime.now()
a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print("running a")
print(a_doubled)
print("---------------------------------------------------------")
print(a)
print("Ran a in: {}".format(str(datetime.datetime.now() - start)))

# alternate invocation -----------------------------------------------------
a = np.random.randn(4, 4)
a = a.astype(np.float32)
a_doubled = np.empty_like(a)
start = datetime.datetime.now()
func(cuda.InOut(a), block=(4, 4, 1))
print("a alternate")
print(a_doubled)
print("---------------------------------------------------------------------")
print(a)
print("Ran a_alt1 in: {}".format(str(datetime.datetime.now() - start)))

# alternate 2 invocation ------------------------------------------------------
import pycuda.gpuarray as gpuarray

start = datetime.datetime.now()
a_gpu = gpuarray.to_gpu(np.random.randn(4, 4).astype(np.float32))
a_doubled = (2*a_gpu).get()
print("a alternate2")
print(a_doubled)
print("---------------------------------------------------------------------")
print(a)
print("Ran a_alt2 in: {}".format(str(datetime.datetime.now() - start)))

# print("a alternate2")
# a = np.random.randn(4, 4)
# a = a.astype(np.float32)
# a_doubled = np.empty_like(a)
# grid = (1, 1)
# block = (4, 4, 1)
# func.prepare("p")
# func.prepared_call(grid, block, a_gpu)
# print(a_doubled)
# print("--------------------------------------------------------------------")
# print(a)
# print("Ran a_alt1 in: {}".format(str(datetime.datetime.now() - start)))

b = np.random.randn(4, 4)
b = b.astype(np.float32)
# b_doubled = np.empty_like(b)

start = datetime.datetime.now()
def doublify(in_array):
    out_array = np.empty_like(in_array)
    for i in range(0, 4):
        out_array[i] = in_array[i][0] + in_array[i][1]*4
    return out_array
b_doubled = doublify(b)
print("running b")
print(b_doubled)
print("-----------------------------------------------------------")
print(b)
print("Ran b in: {}".format(str(datetime.datetime.now() - start)))
