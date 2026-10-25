@echo off
REM NVIDIA Nsight Systems and Nsight Compute automated profiling script for Project GreenSpace

echo =======================================================================
echo   PROJECT GREENSPACE: NVIDIA NSIGHT SYSTEMS & NCU PROFILING TRACE
echo =======================================================================

where nsys >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] 'nsys' (Nsight Systems) not found in PATH.
    echo Running demo with internal PyTorch profiler...
    python scripts/run_demo.py
    exit /b 0
)

echo [1/2] Profiling CUDA Streams and NVTX Ranges with Nsight Systems...
nsys profile ^
    --trace=cuda,nvtx,osrt ^
    --output=traces/greenspace_nsys_trace ^
    --force-overwrite=true ^
    --duration=15 ^
    python scripts/run_demo.py

echo [2/2] Generating summary stats...
nsys stats --report nvtx_sum,cuda_gpu_kern_sum traces/greenspace_nsys_trace.nsys-rep

echo Profile trace successfully saved to traces/greenspace_nsys_trace.nsys-rep

