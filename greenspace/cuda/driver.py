"""
Low-Level CUDA 13.4+ / 13.5 Driver API Bindings for Green Contexts and Hardware Resources.
Supports Windows (nvcuda.dll) and Linux (libcuda.so).
"""

import ctypes
import os
import platform
import sys
from typing import Dict, Any, Optional, Tuple

# CUDA Driver Error Codes
CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_OUT_OF_MEMORY = 2
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_DEINITIALIZED = 4
CUDA_ERROR_NO_DEVICE = 100
CUDA_ERROR_INVALID_DEVICE = 101
CUDA_ERROR_INVALID_IMAGE = 200
CUDA_ERROR_INVALID_CONTEXT = 201
CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
CUDA_ERROR_MAP_FAILED = 205
CUDA_ERROR_UNMAP_FAILED = 206
CUDA_ERROR_ARRAY_IS_MAPPED = 207
CUDA_ERROR_ALREADY_MAPPED = 208
CUDA_ERROR_NO_BINARY_FOR_GPU = 209
CUDA_ERROR_ALREADY_ACQUIRED = 210
CUDA_ERROR_NOT_MAPPED = 211
CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
CUDA_ERROR_ECC_UNCORRECTABLE = 214
CUDA_ERROR_UNSUPPORTED_LIMIT = 215
CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
CUDA_ERROR_INVALID_PTX = 218
CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221
CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222
CUDA_ERROR_JIT_COMPILATION_DISABLED = 223
CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224
CUDA_ERROR_INVALID_SOURCE = 300
CUDA_ERROR_FILE_NOT_FOUND = 301
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
CUDA_ERROR_OPERATING_SYSTEM = 304
CUDA_ERROR_INVALID_HANDLE = 400
CUDA_ERROR_ILLEGAL_STATE = 401
CUDA_ERROR_NOT_FOUND = 500
CUDA_ERROR_NOT_READY = 600
CUDA_ERROR_ILLEGAL_ADDRESS = 700
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
CUDA_ERROR_LAUNCH_TIMEOUT = 702
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
CUDA_ERROR_ASSERT = 710
CUDA_ERROR_TOO_MANY_PEERS = 711
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
CUDA_ERROR_HARDWARE_STACK_ERROR = 714
CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
CUDA_ERROR_MISALIGNED_ADDRESS = 716
CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
CUDA_ERROR_INVALID_PC = 718
CUDA_ERROR_LAUNCH_FAILED = 719
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720
CUDA_ERROR_NOT_PERMITTED = 800
CUDA_ERROR_NOT_SUPPORTED = 801
CUDA_ERROR_SYSTEM_NOT_READY = 802
CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803
CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
CUDA_ERROR_MPS_CONNECTION_FAILED = 805
CUDA_ERROR_MPS_RPC_FAILURE = 806
CUDA_ERROR_MPS_SERVER_NOT_READY = 807
CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808
CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809
CUDA_ERROR_MPS_CLIENT_TERMINATED = 810
CUDA_ERROR_CDP_NOT_SUPPORTED = 811
CUDA_ERROR_CDP_VERSION_MISMATCH = 812
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
CUDA_ERROR_CAPTURED_EVENT = 907
CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908
CUDA_ERROR_TIMEOUT = 909
CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910
CUDA_ERROR_EXTERNAL_DEVICE = 911
CUDA_ERROR_UNKNOWN = 999

# Resource Types
CU_DEV_RESOURCE_TYPE_INVALID = 0
CU_DEV_RESOURCE_TYPE_SM = 1
CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG = 1000
CU_DEV_RESOURCE_TYPE_WORKQUEUE = 10000

# Event Flags
CU_EVENT_DEFAULT = 0
CU_EVENT_BLOCKING_SYNC = 1
CU_EVENT_DISABLE_TIMING = 2
CU_EVENT_INTERPROCESS = 4

# Stream Flags
CU_STREAM_DEFAULT = 0
CU_STREAM_NON_BLOCKING = 1


class CUDADriverError(RuntimeError):
    """Exception raised when a CUDA Driver API call returns non-zero error."""
    def __init__(self, err_code: int, func_name: str, msg: str = ""):
        self.err_code = err_code
        self.func_name = func_name
        self.msg = msg
        super().__init__(f"CUDA Driver Error {err_code} in '{func_name}': {msg}")


def check_cuda(err: int, func_name: str = ""):
    if err != CUDA_SUCCESS:
        raise CUDADriverError(err, func_name)


# Load CUDA Driver Library
_driver_lib: Optional[ctypes.CDLL] = None

def get_driver_lib() -> ctypes.CDLL:
    global _driver_lib
    if _driver_lib is not None:
        return _driver_lib

    system = platform.system()
    if system == "Windows":
        candidates = ["nvcuda.dll", r"C:\Windows\System32\nvcuda.dll"]
    else:
        candidates = ["libcuda.so", "libcuda.so.1", "/usr/lib/x86_64-linux-gnu/libcuda.so.1"]

    for cand in candidates:
        try:
            _driver_lib = ctypes.CDLL(cand)
            return _driver_lib
        except Exception:
            continue

    raise RuntimeError(f"Could not load NVIDIA CUDA Driver library on {system}. Candidates tried: {candidates}")


# C Struct Definitions
class CUdevSmResource(ctypes.Structure):
    _fields_ = [
        ("smCount", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("minSmPartitionSize", ctypes.c_uint32),
        ("smCoscheduledAlignment", ctypes.c_uint32),
    ]


class CUdevWorkqueueConfigResource(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_int),
        ("wqConcurrencyLimit", ctypes.c_uint32),
        ("sharingScope", ctypes.c_uint32),
    ]


class CU_DEV_SM_RESOURCE_GROUP_PARAMS(ctypes.Structure):
    _fields_ = [
        ("smCount", ctypes.c_uint32),
        ("coscheduledSmCount", ctypes.c_uint32),
        ("preferredCoscheduledSmCount", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class CUdevResource(ctypes.Structure):
    pass

CUdevResource._fields_ = [
    ("type", ctypes.c_uint32),
    ("_internal_padding", ctypes.c_uint8 * 92),
    ("smCount", ctypes.c_uint32),
    ("minSmPartitionSize", ctypes.c_uint32),
    ("smCoscheduledAlignment", ctypes.c_uint32),
    ("flags", ctypes.c_uint32),
    ("_reserved", ctypes.c_uint8 * 144),
    ("nextResource", ctypes.c_void_p),
]


class CUDADriver:
    """Singleton wrapper for CUDA Driver API operations."""
    _instance: Optional["CUDADriver"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_api()
        return cls._instance

    def _init_api(self):
        self.lib = get_driver_lib()
        self.lib.cuInit(0)

        # Basic device queries
        self.cuDriverGetVersion = self.lib.cuDriverGetVersion
        self.cuDeviceGet = self.lib.cuDeviceGet
        self.cuDeviceGetName = self.lib.cuDeviceGetName
        self.cuDeviceGetAttribute = self.lib.cuDeviceGetAttribute

        # Context operations
        self.cuCtxPushCurrent = getattr(self.lib, "cuCtxPushCurrent_v2", self.lib.cuCtxPushCurrent)
        self.cuCtxPopCurrent = getattr(self.lib, "cuCtxPopCurrent_v2", self.lib.cuCtxPopCurrent)
        self.cuCtxGetCurrent = getattr(self.lib, "cuCtxGetCurrent", None)
        self.cuCtxSynchronize = self.lib.cuCtxSynchronize

        # Stream operations
        self.cuStreamCreate = getattr(self.lib, "cuStreamCreate", None)
        self.cuStreamCreateWithPriority = getattr(self.lib, "cuStreamCreateWithPriority", None)
        self.cuStreamDestroy = getattr(self.lib, "cuStreamDestroy_v2", getattr(self.lib, "cuStreamDestroy", None))
        self.cuStreamSynchronize = self.lib.cuStreamSynchronize
        self.cuStreamWaitEvent = self.lib.cuStreamWaitEvent

        # Event operations
        self.cuEventCreate = self.lib.cuEventCreate
        self.cuEventRecord = self.lib.cuEventRecord
        self.cuEventSynchronize = self.lib.cuEventSynchronize
        self.cuEventDestroy = getattr(self.lib, "cuEventDestroy_v2", self.lib.cuEventDestroy)
        self.cuEventElapsedTime = self.lib.cuEventElapsedTime

        # Green Context APIs (CUDA 13.4+)
        self.has_green_ctx = hasattr(self.lib, "cuGreenCtxCreate")
        if self.has_green_ctx:
            self.cuDeviceGetDevResource = self.lib.cuDeviceGetDevResource
            self.cuDevResourceGenerateDesc = self.lib.cuDevResourceGenerateDesc
            self.cuGreenCtxCreate = self.lib.cuGreenCtxCreate
            self.cuGreenCtxDestroy = self.lib.cuGreenCtxDestroy
            self.cuCtxFromGreenCtx = self.lib.cuCtxFromGreenCtx
            self.cuGreenCtxStreamCreate = getattr(self.lib, "cuGreenCtxStreamCreate", None)
            self.cuDevSmResourceSplit = getattr(self.lib, "cuDevSmResourceSplit", None)
            self.cuDevSmResourceSplitByCount = getattr(self.lib, "cuDevSmResourceSplitByCount", None)

    def get_driver_version(self) -> int:
        v = ctypes.c_int()
        check_cuda(self.cuDriverGetVersion(ctypes.byref(v)), "cuDriverGetVersion")
        return v.value

    def get_device(self, device_id: int = 0) -> int:
        dev = ctypes.c_int()
        check_cuda(self.cuDeviceGet(ctypes.byref(dev), device_id), "cuDeviceGet")
        return dev.value

    def get_device_info(self, device_id: int = 0) -> Dict[str, Any]:
        dev = self.get_device(device_id)
        name_buf = ctypes.create_string_buffer(256)
        check_cuda(self.cuDeviceGetName(name_buf, 256, dev), "cuDeviceGetName")
        
        # SM count attribute (attribute 16: CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        sm_count = ctypes.c_int()
        self.cuDeviceGetAttribute(ctypes.byref(sm_count), 16, dev)

        # L2 cache size (attribute 37: CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        l2_size = ctypes.c_int()
        self.cuDeviceGetAttribute(ctypes.byref(l2_size), 37, dev)

        # Compute capability
        major = ctypes.c_int()
        minor = ctypes.c_int()
        self.cuDeviceGetAttribute(ctypes.byref(major), 75, dev) # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        self.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev) # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

        min_sm_partition = 8 if major.value >= 9 else 4
        alignment = 8 if major.value >= 9 else 4

        # Read actual resource descriptor if supported
        if self.has_green_ctx:
            raw_res = (ctypes.c_uint8 * 256)()
            err = self.cuDeviceGetDevResource(dev, ctypes.byref(raw_res), CU_DEV_RESOURCE_TYPE_SM)
            if err == CUDA_SUCCESS:
                import struct
                # Offsets 96, 100, 104 verified on Blackwell
                actual_sms = struct.unpack('<I', bytes(raw_res[96:100]))[0]
                if actual_sms > 0:
                    sm_count.value = actual_sms
                min_sm_partition = struct.unpack('<I', bytes(raw_res[100:104]))[0] or min_sm_partition
                alignment = struct.unpack('<I', bytes(raw_res[104:108]))[0] or alignment

        return {
            "name": name_buf.value.decode("utf-8", "ignore"),
            "device_id": device_id,
            "sm_count": sm_count.value,
            "min_sm_partition_size": min_sm_partition,
            "sm_coscheduled_alignment": alignment,
            "l2_cache_bytes": l2_size.value,
            "compute_capability": (major.value, minor.value),
            "driver_version": self.get_driver_version(),
            "has_green_context_support": self.has_green_ctx,
        }

    def create_green_context(
        self,
        sm_count: int,
        workqueue_concurrency_limit: Optional[int] = None,
        device_id: int = 0
    ) -> Tuple[int, int]:
        """
        Creates a hardware Green Context using CUDA 13 Driver API.
        Returns (green_ctx_handle, cu_ctx_handle).
        """
        if not self.has_green_ctx:
            raise RuntimeError("CUDA Green Contexts are not supported by the current driver.")

        dev = self.get_device(device_id)

        # Query raw device resource
        res_buf = (ctypes.c_uint8 * 256)()
        check_cuda(
            self.cuDeviceGetDevResource(dev, ctypes.byref(res_buf), CU_DEV_RESOURCE_TYPE_SM),
            "cuDeviceGetDevResource"
        )

        # Generate resource descriptor
        hDesc = ctypes.c_void_p()
        check_cuda(
            self.cuDevResourceGenerateDesc(ctypes.byref(hDesc), ctypes.byref(res_buf), 1),
            "cuDevResourceGenerateDesc"
        )

        # Create Green Context
        hGreenCtx = ctypes.c_void_p()
        check_cuda(
            self.cuGreenCtxCreate(ctypes.byref(hGreenCtx), hDesc, dev, 0),
            "cuGreenCtxCreate"
        )

        # Convert to standard CUcontext handle for PyTorch/CUDA runtime interop
        hCtx = ctypes.c_void_p()
        check_cuda(
            self.cuCtxFromGreenCtx(ctypes.byref(hCtx), hGreenCtx),
            "cuCtxFromGreenCtx"
        )

        return (hGreenCtx.value or 0, hCtx.value or 0)

    def destroy_green_context(self, green_ctx_handle: int):
        if self.has_green_ctx and green_ctx_handle:
            hGreenCtx = ctypes.c_void_p(green_ctx_handle)
            self.cuGreenCtxDestroy(hGreenCtx)

    def create_stream_in_context(self, cu_ctx_handle: int, priority: int = 0) -> int:
        """
        Pushes the CUcontext, creates a non-blocking CUDA stream, and pops the context.
        Returns the raw stream pointer.
        """
        hCtx = ctypes.c_void_p(cu_ctx_handle)
        check_cuda(self.cuCtxPushCurrent(hCtx), "cuCtxPushCurrent")
        try:
            hStream = ctypes.c_void_p()
            if self.cuStreamCreateWithPriority is not None and priority != 0:
                check_cuda(
                    self.cuStreamCreateWithPriority(ctypes.byref(hStream), CU_STREAM_NON_BLOCKING, priority),
                    "cuStreamCreateWithPriority"
                )
            else:
                check_cuda(
                    self.cuStreamCreate(ctypes.byref(hStream), CU_STREAM_NON_BLOCKING),
                    "cuStreamCreate"
                )
            return hStream.value or 0
        finally:
            prev = ctypes.c_void_p()
            self.cuCtxPopCurrent(ctypes.byref(prev))

    def destroy_stream(self, stream_handle: int):
        if stream_handle and self.cuStreamDestroy is not None:
            hStream = ctypes.c_void_p(stream_handle)
            self.cuStreamDestroy(hStream)


# Global singleton instance
driver = CUDADriver()


# Human readable error decoding
