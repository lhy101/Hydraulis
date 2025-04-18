#include "hydraulis/impl/utils/cuda_utils.h"

namespace hydraulis {
namespace cuda {

namespace {
thread_local int current_device_id = -1;
} // namespace

#if CUDA_VERSION >= 12000
void CudaTryGetDevice(int* device_id) {
  *device_id = current_device_id;
}

void CudaSetDevice(int device_id) {
  // HT_LOG_INFO << "device id is " << device_id << " and old device id is " << current_device_id;
  if (current_device_id != device_id) {
    CUDA_CALL(cudaSetDevice(device_id));
    current_device_id = device_id;
  }
}
#endif

NDArray to_int64_ndarray(const std::vector<int64_t>& vec,
                         DeviceIndex device_id) {
  auto ret = NDArray::empty({static_cast<int64_t>(vec.size())},
                            Device(kCUDA, device_id), kInt64, kBlockingStream);
  hydraulis::cuda::CUDADeviceGuard guard(device_id);
  CudaMemcpy(ret->raw_data_ptr(), vec.data(), vec.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  return ret;
}

NDArray to_int64_ndarray(const int64_t* from, size_t n, DeviceIndex device_id) {
  auto ret = NDArray::empty({static_cast<int64_t>(n)}, Device(kCUDA, device_id),
                            kInt64, kBlockingStream);
  CudaMemcpy(ret->raw_data_ptr(), from, n * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  return ret;
}

NDArray to_byte_ndarray(const std::vector<uint8_t>& vec,
                        DeviceIndex device_id) {
  auto ret = NDArray::empty({static_cast<int64_t>(vec.size())},
                            Device(kCUDA, device_id), kByte, kBlockingStream);
  hydraulis::cuda::CUDADeviceGuard guard(device_id);
  CudaMemcpy(ret->raw_data_ptr(), vec.data(), vec.size() * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
  return ret;
}

NDArray to_byte_ndarray(const uint8_t* from, size_t n, DeviceIndex device_id) {
  auto ret = NDArray::empty({static_cast<int64_t>(n)}, Device(kCUDA, device_id),
                            kByte, kBlockingStream);
  CudaMemcpy(ret->raw_data_ptr(), from, n * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
  return ret;
}

} // namespace cuda
} // namespace hydraulis
