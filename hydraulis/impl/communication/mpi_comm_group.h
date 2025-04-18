#pragma once

#include "hydraulis/impl/communication/comm_group.h"
#include <mpi.h>
#include <queue>
#include <future>

namespace hydraulis {
namespace impl {
namespace comm {

class MPICommunicationGroupDef;
class MPICommunicationGroup;

class MPICommunicationGroupDef : public CommunicationGroupDef {
 protected:
  friend class MPICommunicationGroup;
  struct constructor_access_key {};
  MPICommunicationGroupDef(const std::vector<int>& world_ranks,
                           const Stream& stream);

 public:
  MPICommunicationGroupDef(const constructor_access_key&,
                           const std::vector<int>& world_ranks,
                           const Stream& stream)
  : MPICommunicationGroupDef(world_ranks, stream) {}

  MPICommunicationGroupDef(const constructor_access_key&,
                           const std::vector<int>& world_ranks)
  : MPICommunicationGroupDef(
      world_ranks,
      Stream(kCPU, world_ranks.size() != 2 ? kCollectiveStream : kP2PStream)) {}

  ~MPICommunicationGroupDef();

  void Broadcast(NDArray& data, int broadcaster) override;

  void AllReduce(const NDArray& input, NDArray& output,
                 ReductionType red_type = kSUM) override;

  void AllReduceCoalesce(const NDArrayList& inputs, NDArrayList& outputs,
                         NDArray contiguous_buffers,
                         ReductionType red_type = kSUM) override;

  void AlltoAll(const NDArray& input, NDArray& output) override;

  void Reduce(const NDArray& input, NDArray& output, int reducer,
              ReductionType red_type = kSUM) override;

  void AllGather(const NDArray& input, NDArray& output, int32_t gather_dim = 0) override;

  void ReduceScatter(const NDArray& input, NDArray& output,
                     int32_t scatter_dim = 0, ReductionType red_type = kSUM) override;

  void Gather(const NDArray& input, NDArray& output, int gatherer) override;

  void Scatter(const NDArray& input, NDArray& output, int scatterer) override;

  void Send(const NDArray& data, int receiver) override;

  void Recv(NDArray& data, int sender) override;

  CommTask ISend(const NDArray& data, int receiver) override;

  CommTask IRecv(NDArray& data, int sender) override;

  void BatchedISendIRecv(const std::vector<CommTask>& tasks) override;
  
  void Barrier(bool sync = false) override;

  void Sync() override;

  std::string backend() const override {
    return "MPI";
  }

 protected:
  static void WaitRequest(MPI_Request request);

  MPI_Comm _comm{MPI_COMM_NULL};
  std::future<void> _latest_future;
};

class MPICommunicationGroup final
: public CommGroupWrapper<MPICommunicationGroupDef> {
 protected:
  MPICommunicationGroup(const std::vector<int>& world_ranks)
  : CommGroupWrapper<MPICommunicationGroupDef>(
      make_ptr<MPICommunicationGroupDef>(
        MPICommunicationGroupDef::constructor_access_key(), world_ranks)) {}

  MPICommunicationGroup(const std::vector<int>& world_ranks,
                        const Stream& stream)
  : CommGroupWrapper<MPICommunicationGroupDef>(
      make_ptr<MPICommunicationGroupDef>(
        MPICommunicationGroupDef::constructor_access_key(), world_ranks,
        stream)) {}

 public:
  MPICommunicationGroup() = default;

  static MPICommunicationGroup& GetOrCreate(const std::vector<int>& world_ranks,
                                            const Stream& stream);

  static MPICommunicationGroup& GetOrCreate(const std::vector<int>& world_ranks,
                                            Device device = {kCPU}) {
    return GetOrCreate(
      world_ranks,
      Stream(device, world_ranks.size() != 2 ? kCollectiveStream : kP2PStream));
  }

  static MPICommunicationGroup& GetOrCreateWorldwide(const Stream& stream);

  static MPICommunicationGroup& GetOrCreateWorldwide(Device device = {kCPU}) {
    return GetOrCreateWorldwide(Stream(device, kCollectiveStream));
  }
};

int GetMPIWorldRank();
int GetMPIWorldSize();
int GetMPIGroupRank(const std::vector<int>& world_ranks);
void MPISetUpDeviceMappingWithAssignedLocalDeviceOnce(const Device& local_device);
Device MPISetUpDeviceMappingAndAssignLocalDeviceOnce(
  const std::map<DeviceType, int>& resources = {{kCUDA, 8}},
  const std::vector<int64_t>& device_idxs = {});

} // namespace comm
} // namespace impl
} // namespace hydraulis