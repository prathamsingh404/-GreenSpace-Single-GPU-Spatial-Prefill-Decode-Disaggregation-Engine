// Standalone hardware diagnostic. Compile the kernel with the target driver's JIT.
#include <cuda.h>
#include <nvrtc.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "probe_source.h"

static void check(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(status, &name);
        throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA error"));
    }
}
#define CU_CHECK(call) check((call), #call)

static void check_rtc(nvrtcResult status) {
    if (status != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(status));
}

struct Partition {
    CUgreenCtx context{};
    CUstream stream{};
    CUdeviceptr output{};
    ~Partition() {
        if (stream) cuStreamSynchronize(stream);
        if (output) cuMemFree(output);
        if (stream) cuStreamDestroy(stream);
        if (context) cuGreenCtxDestroy(context);
    }
    void init(CUdevice dev, CUdevResource& resource, unsigned int blocks) {
        CUdevResourceDesc desc{};
        CU_CHECK(cuDevResourceGenerateDesc(&desc, &resource, 1));
        CU_CHECK(cuGreenCtxCreate(&context, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
        CU_CHECK(cuGreenCtxStreamCreate(&stream, context, CU_STREAM_NON_BLOCKING, 0));
        CU_CHECK(cuMemAlloc(&output, blocks * sizeof(int)));
    }
    std::set<int> read(unsigned int blocks) const {
        std::vector<int> ids(blocks);
        CU_CHECK(cuStreamSynchronize(stream));
        CU_CHECK(cuMemcpyDtoH(ids.data(), output, blocks * sizeof(int)));
        return {ids.begin(), ids.end()};
    }
};

static void probe(CUdevice dev) {
    constexpr unsigned int blocks = 1024;
    CUdevResource full{}, decode{}, prefill{};
    CU_CHECK(cuDeviceGetDevResource(dev, &full, CU_DEV_RESOURCE_TYPE_SM));
    unsigned int groups = 1;
    CU_CHECK(cuDevSmResourceSplitByCount(&decode, &groups, &full, &prefill, 0,
                                         std::max(1u, full.sm.smCount * 2 / 5)));
    if (groups != 1 || !prefill.sm.smCount) throw std::runtime_error("No complementary split");
    Partition p, d;
    p.init(dev, prefill, blocks);
    d.init(dev, decode, blocks);

    int major{}, minor{};
    CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
    const char* options[] = {arch.c_str()};
    nvrtcProgram program{};
    check_rtc(nvrtcCreateProgram(&program, kernel_source, "record_smids.cu", 0, nullptr, nullptr));
    check_rtc(nvrtcCompileProgram(program, 1, options));
    size_t size{};
    check_rtc(nvrtcGetPTXSize(program, &size));
    std::vector<char> ptx(size);
    check_rtc(nvrtcGetPTX(program, ptx.data()));
    check_rtc(nvrtcDestroyProgram(&program));
    CUmodule module{};
    CUfunction kernel{};
    CU_CHECK(cuModuleLoadData(&module, ptx.data()));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "record_smids_kernel"));
    for (auto part : {&p, &d}) {
        void* args[] = {&part->output};
        CU_CHECK(cuLaunchKernel(kernel, blocks, 1, 1, 32, 1, 1, 0, part->stream, args, nullptr));
    }
    auto pids = p.read(blocks), dids = d.read(blocks);
    CU_CHECK(cuModuleUnload(module));
    for (auto* set : {&pids, &dids}) {
        std::cout << (set == &pids ? "prefill:" : "decode:");
        for (int id : *set) std::cout << " " << id;
        std::cout << "\n";
    }
    if (pids.size() > prefill.sm.smCount || dids.size() > decode.sm.smCount ||
        std::any_of(dids.begin(), dids.end(), [&](int id) { return pids.count(id); }))
        throw std::runtime_error("Observed SMs violate the driver split");
}

int main(int argc, char** argv) try {
    CU_CHECK(cuInit(0));
    CUdevice dev{};
    CU_CHECK(cuDeviceGet(&dev, argc > 1 ? std::stoi(argv[1]) : 0));
    CUcontext primary{};
    CU_CHECK(cuDevicePrimaryCtxRetain(&primary, dev));
    CU_CHECK(cuCtxSetCurrent(primary));
    probe(dev);
    CU_CHECK(cuCtxSetCurrent(nullptr));
    CU_CHECK(cuDevicePrimaryCtxRelease(dev));
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return EXIT_FAILURE;
}
