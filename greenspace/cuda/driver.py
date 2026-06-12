"""Low-level CUDA Driver API Loader."""
import os
import ctypes
import platform

def load_cuda_lib():
    dll = "nvcuda.dll" if platform.system() == "Windows" else "libcuda.so"
    return ctypes.CDLL(dll)

CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_OUT_OF_MEMORY = 2

class CUDADriverError(RuntimeError):
    pass

class CUdevSmResource(ctypes.Structure):
    _fields_ = [("smCount", ctypes.c_uint32)]

class CUdevResource(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("sm", CUdevSmResource)
    ]

CUgreenCtx = ctypes.c_void_p
CUcontext = ctypes.c_void_p
CUstream = ctypes.c_void_p

# cuDeviceGetDevResource driver binding

# cuDevResourceGenerateDesc driver binding
