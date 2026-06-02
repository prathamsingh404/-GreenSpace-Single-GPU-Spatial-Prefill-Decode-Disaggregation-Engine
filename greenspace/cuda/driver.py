"""Low-level CUDA Driver API Loader."""
import os
import ctypes
import platform

def load_cuda_lib():
    dll = "nvcuda.dll" if platform.system() == "Windows" else "libcuda.so"
    return ctypes.CDLL(dll)
