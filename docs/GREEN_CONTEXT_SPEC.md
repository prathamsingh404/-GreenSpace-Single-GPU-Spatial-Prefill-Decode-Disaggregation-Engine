# CUDA 13.4+ / 13.5 Green Context API Specification
## Low-Level Driver Architecture & Systems Reference

---

## 1. Introduction

CUDA Green Contexts (`CUgreenCtx`) introduce lightweight spatial resource partitioning directly into the NVIDIA CUDA Driver API. Unlike traditional CUDA contexts (`CUcontext`), which require substantial driver virtualization overhead and multiplex all hardware across a single monolithic execution unit, Green Contexts partition hardware resources within a single process.

---

## 2. Driver Structures & Memory Layouts

### 2.1 Hardware Resource Descriptor (`CUdevResource`)
The fundamental building block is `CUdevResource`, a tagged union describing physical hardware allocations:

```c
typedef enum CUdevResourceType_enum {
    CU_DEV_RESOURCE_TYPE_INVALID          = 0,
    CU_DEV_RESOURCE_TYPE_SM               = 1,
    CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG = 1000,
    CU_DEV_RESOURCE_TYPE_WORKQUEUE        = 10000
} CUdevResourceType;

struct CUdevSmResource {
    unsigned int smCount;                  // Number of SMs allocated
    unsigned int flags;                    // Resource configuration flags
    unsigned int minSmPartitionSize;       // Architecture minimum partition size
    unsigned int smCoscheduledAlignment;   // Co-scheduling alignment constraint
};

struct CUdevWorkqueueConfigResource {
    CUdevice device;                       // Target CUDA device handle
    unsigned int wqConcurrencyLimit;       // Max concurrent stream-ordered workloads
    unsigned int sharingScope;             // Scope of workqueue sharing
};

typedef struct CUdevResource_st {
    CUdevResourceType type;                // Offset 0 (4 bytes)
    unsigned char _internal_padding[92];   // Offset 4 to 96
    union {
        CUdevSmResource sm;                // Offset 96: smCount, minSmPartitionSize, ...
        CUdevWorkqueueConfigResource wqConfig;
        CUdevWorkqueueResource wq;
    };
    struct CUdevResource_st *nextResource; // Offset 248: Linked list pointer
} CUdevResource;
```

---

## 3. Driver API Lifecycle Functions

### 3.1 Device Resource Discovery
```c
CUresult cuDeviceGetDevResource(
    CUdevice dev, 
    CUdevResource *resource, 
    CUdevResourceType type
);
```
Queries available device resources for a physical GPU device handle.

### 3.2 Descriptor Generation
```c
CUresult cuDevResourceGenerateDesc(
    CUdevResourceDesc *phDesc, 
    CUdevResource *resources, 
    unsigned int nbResources
);
```
Packages an array of partitioned resource descriptors into an opaque hardware configuration descriptor.

### 3.3 Green Context Creation & Destruction
```c
CUresult cuGreenCtxCreate(
    CUgreenCtx *phCtx, 
    CUdevResourceDesc desc, 
    CUdevice dev, 
    unsigned int flags
);

CUresult cuGreenCtxDestroy(CUgreenCtx hCtx);
```

### 3.4 Context Bridging & Stream Allocation
```c
// Converts lightweight CUgreenCtx into standard CUcontext
CUresult cuCtxFromGreenCtx(
    CUcontext *pContext, 
    CUgreenCtx hCtx
);

// Creates custom stream targeted strictly to the Green Context partition
CUresult cuGreenCtxStreamCreate(
    CUstream *phStream, 
    CUgreenCtx greenCtx, 
    unsigned int flags
);
```

---

## 4. Hardware Granularity & Architectural Minimums

| Compute Capability | Architecture | Min Partition Size | Co-Scheduled Alignment |
|---|---|---|---|
| **SM 7.0 - 7.5** | Volta / Turing | 2 SMs | 2 SMs |
| **SM 8.0 - 8.9** | Ampere / Ada Lovelace | 4 SMs | 4 SMs |
| **SM 9.0 - 9.0a** | Hopper | 8 SMs | 8 SMs |
| **SM 12.0+** | Blackwell (e.g. RTX 50-Series) | 8 SMs | 8 SMs |

On NVIDIA Blackwell GPUs (such as the RTX 5050 Laptop GPU with 20 total SMs), valid spatial partitions are carved in multiples of 8 SMs (e.g., 12 SMs Prefill, 8 SMs Decode).

