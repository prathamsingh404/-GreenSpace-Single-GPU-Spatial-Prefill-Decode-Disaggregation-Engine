"""Checked CUDA Driver API calls using NVIDIA's versioned resource structures."""

from typing import Any
from uuid import UUID

from cuda.bindings import driver as cu


class CUDADriverError(RuntimeError):
    def __init__(self, result, operation: str):
        self.err_code = int(result)
        super().__init__(f"{operation}: {result.name} ({self.err_code})")


def checked(function, *args):
    """Check CUresult before exposing any driver output."""
    result, *values = function(*args)
    if result != cu.CUresult.CUDA_SUCCESS:
        raise CUDADriverError(result, function.__name__)
    if len(values) == 1:
        return values[0]
    return tuple(values)


class CUDADriver:
    def get_driver_version(self) -> int:
        return checked(cu.cuDriverGetVersion)

    def get_device(self, device_id: int = 0):
        checked(cu.cuInit, 0)
        return checked(cu.cuDeviceGet, device_id)

    def sm_resource(self, device_id: int):
        return checked(cu.cuDeviceGetDevResource, self.get_device(device_id),
                       cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)

    def get_device_info(self, device_id: int = 0) -> dict[str, Any]:
        dev = self.get_device(device_id)
        attrs = cu.CUdevice_attribute
        def attr(key):
            return checked(cu.cuDeviceGetAttribute, key, dev)

        info = {
            "name": checked(cu.cuDeviceGetName, 256, dev).split(b"\0")[0].decode(),
            "uuid": str(UUID(bytes=checked(cu.cuDeviceGetUuid, dev).bytes)),
            "device_id": device_id,
            "sm_count": attr(attrs.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            "l2_cache_bytes": attr(attrs.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
            "compute_capability": (attr(attrs.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                                   attr(attrs.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)),
            "driver_version": self.get_driver_version(),
            "has_green_context_support": False,
            "min_sm_partition_size": None,
            "sm_coscheduled_alignment": None,
            "locality_domain_id": None,
        }
        try:
            resource = self.sm_resource(device_id)
        except (CUDADriverError, RuntimeError) as error:
            info["resource_query_error"] = str(error)
            return info
        sm = resource.sm
        info.update(has_green_context_support=True,
                    min_sm_partition_size=sm.minSmPartitionSize or None,
                    sm_coscheduled_alignment=sm.smCoscheduledAlignment or None)
        if sm.flags & int(cu.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID):
            info["locality_domain_id"] = sm.localityDomainId
        return info

    def split(self, device_id: int, decode_sms: int):
        resource = self.sm_resource(device_id)
        if not 0 < decode_sms < resource.sm.smCount:
            raise ValueError("A split requires a positive SM count smaller than the device")
        groups, count, remainder = checked(cu.cuDevSmResourceSplitByCount,
                                            1, resource, 0, decode_sms)
        if count != 1 or not remainder.sm.smCount:
            raise ValueError(f"Driver cannot leave a complementary partition for {decode_sms} SMs")
        return remainder, groups[0]

    def create_green_context(self, resource, device_id: int, workqueue_limit: int | None):
        dev = self.get_device(device_id)
        resources = [resource]
        if workqueue_limit is not None:
            if workqueue_limit < 1:
                raise ValueError("workqueue limit must be positive")
            wq = checked(cu.cuDeviceGetDevResource, dev,
                         cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG)
            wq.wqConfig.wqConcurrencyLimit = min(workqueue_limit, wq.wqConfig.wqConcurrencyLimit)
            wq.wqConfig.sharingScope = cu.CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED
            resources.append(wq)
        desc = checked(cu.cuDevResourceGenerateDesc, resources, len(resources))
        handle = checked(cu.cuGreenCtxCreate, desc, dev,
                         int(cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        return handle, resources[-1].wqConfig.wqConcurrencyLimit if workqueue_limit is not None else None


driver = CUDADriver()
