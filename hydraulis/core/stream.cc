#include "hydraulis/core/stream.h"
#include "hydraulis/impl/stream/CPUStream.h"
#include "hydraulis/impl/stream/CUDAStream.h"

namespace hydraulis {

void Stream::Sync() const {
  if (_device.is_cpu()) {
    hydraulis::impl::CPUStream(*this).Sync();
  } else if (_device.is_cuda()) {
    hydraulis::impl::CUDAStream(*this).Sync();
  }
}

std::ostream& operator<<(std::ostream& os, const Stream& stream) {
  os << "stream(" << stream.device()
     << ", stream_index=" << stream.stream_index() << ")";
  return os;
}

void SynchronizeAllStreams(const Device& device) {
  if (device.is_cpu()) {
    hydraulis::impl::SynchronizeAllCPUStreams();
  } else if (device.is_cuda()) {
    hydraulis::impl::SynchronizeAllCUDAStreams(device);
  } else {
    hydraulis::impl::SynchronizeAllCPUStreams();
    hydraulis::impl::SynchronizeAllCUDAStreams(device);
  }
}

} // namespace hydraulis
