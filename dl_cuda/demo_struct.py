import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
struct DoubleOperation {
    int datalen, __padding; // so 64-bit ptrs can be aligned
    float *ptr;
};
""")

class DoubleOpStruct:
    mem_size = 8 + np.intp(0).nbytes
    def __init__(self, array, struct_arr_ptr):
        self.data = cuda.to_device(array)
        self.shape, self.dtype = array.shape, array.dtype
        cuda.memcpy_htod(int(struct_arr_ptr), np.frombuffer(np.int32(array.size)))
        cuda.memcpy_htod(int(struct_arr_ptr) + 8, np.frombuffer(np.intp(int(self.data))))

    def __str__(self):
        return str(cuda.from_device(self.data, self.shape, self.dtype))

struct_arr = cuda.mem_alloc(2 * DoubleOpStruct.mem_size)
do2_ptr = int(struct_arr) + DoubleOpStruct.mem_size

array1 = DoubleOpStruct(np.array([1,2,3], dtype=np.float32), struct_arr)
array2 = DoubleOpStruct(np.array([0,4], dtype=np.float32), do2_ptr)
print("original arrays", array1, array2)

func = mod.get_function("double_array")
func(struct_arr, block=(32, 1, 1), grid=(2, 1))
print("doubled arrays:", array1, array2)

func(np.intp(do2_ptr), block=(32, 1, 1), grid=(1, 1))
print("doubled second only", array1, array2)
