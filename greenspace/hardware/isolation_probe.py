"""On-device verification of complementary Green Context stream placement."""

import ctypes
from pathlib import Path

import torch
from cuda.bindings import driver as cu, nvrtc

from greenspace.cuda.driver import checked


def compile_smid_kernel():
    """Compile and load the diagnostic kernel on the current CUDA device."""
    def rtc(function, *args):
        status, *values = function(*args)
        if int(status):
            raise RuntimeError(f"{function.__name__}: {status.name}")
        return values[0] if values else None

    source = Path(__file__).with_name("record_smids.cuh").read_bytes()
    program = rtc(nvrtc.nvrtcCreateProgram, source, b"smids.cu", 0, [], [])
    try:
        major, minor = torch.cuda.get_device_capability()
        rtc(nvrtc.nvrtcCompileProgram, program, 1, [f"--gpu-architecture=compute_{major}{minor}".encode()])
        ptx = bytearray(rtc(nvrtc.nvrtcGetPTXSize, program))
        rtc(nvrtc.nvrtcGetPTX, program, ptx)
        return checked(cu.cuModuleLoadData, bytes(ptx))
    finally:
        rtc(nvrtc.nvrtcDestroyProgram, program)


def observed_smids(kernel, streams):
    """Read every output slot back after both diagnostic launches finish."""
    outputs = []
    for stream in streams:
        with torch.cuda.stream(stream):
            output = torch.full((1024,), -1, device=stream.device, dtype=torch.int32)
        checked(cu.cuLaunchKernel, kernel, 1024, 1, 1, 32, 1, 1, 0,
                cu.CUstream(stream.cuda_stream), ((output.data_ptr(),), (ctypes.c_void_p,)), 0)
        outputs.append(output)
    for stream in streams:
        stream.synchronize()
    return [sorted(set(output.tolist())) for output in outputs]



def verify_pairs(lattice, device_id: int) -> list[dict]:
    """Exercise the exact production stream handles before admission."""
    rows = []
    with torch.cuda.device(device_id):
        module = compile_smid_kernel()
        try:
            kernel = checked(cu.cuModuleGetFunction, module, b"record_smids_kernel")
            for pair in lattice.pairs.values():
                p, d = observed_smids(kernel, (pair.prefill_stream.torch_stream, pair.decode_stream.torch_stream))
                passed = (0 < len(p) <= pair.prefill.num_sms and 0 < len(d) <= pair.decode.num_sms
                          and min(p + d) >= 0 and set(p).isdisjoint(d))
                if not passed:
                    raise RuntimeError(f"Physical SM isolation check failed: prefill={p}, decode={d}")
                rows.append(dict(prefill_observed=p, decode_observed=d, prefill_sms=pair.prefill.num_sms,
                                 decode_sms=pair.decode.num_sms, passed=passed))
        finally:
            checked(cu.cuModuleUnload, module)
    return rows
