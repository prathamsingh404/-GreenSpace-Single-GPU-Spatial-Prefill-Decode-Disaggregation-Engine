extern "C" __global__ void record_smids_kernel(int *output) {
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        output[blockIdx.x] = static_cast<int>(smid);
    }
}
