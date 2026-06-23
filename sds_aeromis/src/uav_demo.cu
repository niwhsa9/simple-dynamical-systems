#include <aeromis_core/dynamical_systems/aerial_vehicle.hpp>
#include <iostream>

#include "aeromis_core/util/array.hpp"
#include "sds_core.cuh"

template <typename Scalar>
class AerialVehicleManager
{
 public:
  using ScalarType = Scalar;

  __host__ AerialVehicleWrapper(
      aeromis_core::AerialVehicleDescription<aeromis_core::GPU, Scalar>&&
          description,
      std::optional<std::string> maybe_flow_field_path = std::nullopt,
      const std::optional<std::vector<Scalar>>& noise_std = std::nullopt)
      : aerial_vehicle_(
            new aeromis_core::AerialVehicle<aeromis_core::GPU, Scalar>(
                std::move(description), maybe_flow_field_path, noise_std))
  {
    aerial_vehicle_->copy_members_to_gpu(aerial_vehicle_->get_gpu_instance());
    cudaMallocManaged(
        &gpu_memory_buffer, aerial_vehicle_->required_bytes_per_thread());
    aerial_vehicle_gpu_ = aerial_vehicle_->get_gpu_instance();
  }

  __device__ __host__ ~AerialVehicleWrapper()
  {
    printf("AerialVehicleWrapper destructor called\n");
#ifdef __CUDA_ARCH__
    printf("Freeing GPU memory buffer\n");
#endif
    // #ifndef __CUDA_ARCH__
    //     cudaFree(gpu_memory_buffer);
    //     delete aerial_vehicle_;
    // #endif
  }

  __device__ __host__ void dynamics(
      Scalar t, const Scalar* x, const Scalar* u, Scalar* dxdt)
  {
#ifdef __CUDA_ARCH__
    int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    aeromis_core::Array<aeromis_core::GPU, Scalar> x_view(
        const_cast<Scalar*>(x), aerial_vehicle_->get_n_x());
    aeromis_core::Array<aeromis_core::GPU, Scalar> u_view(
        const_cast<Scalar*>(u), aerial_vehicle_->get_n_u());
    aeromis_core::Array<aeromis_core::GPU, Scalar> dxdt_view(
        dxdt, aerial_vehicle_->get_n_x());
    aerial_vehicle_gpu_->dynamics(
        t, x_view, u_view, dxdt_view, batch_idx,
        gpu_memory_buffer);  // batch_idx = 0, memory_buffer = nullptr
#endif
  }

  __host__ __device__ int get_n_x() const { return aerial_vehicle_->get_n_x(); }

  __host__ __device__ int get_n_u() const { return aerial_vehicle_->get_n_u(); }

  __host__ __device__ AerialVehicleWrapper(const AerialVehicleWrapper&)
  {
    printf("AerialVehicleWrapper copy constructor called\n");
  }

 private:
  aeromis_core::AerialVehicle<aeromis_core::GPU, Scalar>* aerial_vehicle_;
  aeromis_core::AerialVehicle<aeromis_core::GPU, Scalar>* aerial_vehicle_gpu_;
  uint8_t* gpu_memory_buffer;
};

template <typename Scalar>
class AerialVehicleWrapper
{
 public:
  using ScalarType = Scalar;

  __host__ AerialVehicleWrapper(
      aeromis_core::AerialVehicleDescription<aeromis_core::GPU, Scalar>&&
          description,
      std::optional<std::string> maybe_flow_field_path = std::nullopt,
      const std::optional<std::vector<Scalar>>& noise_std = std::nullopt)
      : aerial_vehicle_(
            new aeromis_core::AerialVehicle<aeromis_core::GPU, Scalar>(
                std::move(description), maybe_flow_field_path, noise_std))
  {
    aerial_vehicle_->copy_members_to_gpu(aerial_vehicle_->get_gpu_instance());
    cudaMallocManaged(
        &gpu_memory_buffer, aerial_vehicle_->required_bytes_per_thread());
    aerial_vehicle_gpu_ = aerial_vehicle_->get_gpu_instance();
  }

  __device__ __host__ ~AerialVehicleWrapper()
  {
    printf("AerialVehicleWrapper destructor called\n");
#ifdef __CUDA_ARCH__
    printf("Freeing GPU memory buffer\n");
#endif
    // #ifndef __CUDA_ARCH__
    //     cudaFree(gpu_memory_buffer);
    //     delete aerial_vehicle_;
    // #endif
  }

  __device__ __host__ void dynamics(
      Scalar t, const Scalar* x, const Scalar* u, Scalar* dxdt)
  {
#ifdef __CUDA_ARCH__
    int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    aeromis_core::Array<aeromis_core::GPU, Scalar> x_view(
        const_cast<Scalar*>(x), aerial_vehicle_->get_n_x());
    aeromis_core::Array<aeromis_core::GPU, Scalar> u_view(
        const_cast<Scalar*>(u), aerial_vehicle_->get_n_u());
    aeromis_core::Array<aeromis_core::GPU, Scalar> dxdt_view(
        dxdt, aerial_vehicle_->get_n_x());
    aerial_vehicle_gpu_->dynamics(
        t, x_view, u_view, dxdt_view, batch_idx,
        gpu_memory_buffer);  // batch_idx = 0, memory_buffer = nullptr
#endif
  }

  __host__ __device__ int get_n_x() const { return aerial_vehicle_->get_n_x(); }

  __host__ __device__ int get_n_u() const { return aerial_vehicle_->get_n_u(); }

  __host__ __device__ AerialVehicleWrapper(const AerialVehicleWrapper&)
  {
    printf("AerialVehicleWrapper copy constructor called\n");
  }

 private:
  aeromis_core::AerialVehicle<aeromis_core::GPU, Scalar>* aerial_vehicle_;
  aeromis_core::AerialVehicle<aeromis_core::GPU, Scalar>* aerial_vehicle_gpu_;
  uint8_t* gpu_memory_buffer;
};

int main()
{
  aeromis_core::AerialVehicleDescription<aeromis_core::GPU, float> description(
      "/home/ashwin/sources/aeromis/src/aeromis_simulator/urdf/edge540_24/"
      "edge540_24.uardf");
  AerialVehicleWrapper<float> wrapper(std::move(description));
  sds::RK2 integrator;

  Tensor<float, 1> x0(wrapper.get_n_x());
  x0(3) = 1.0;
  x0(7) = 7.0;

  Tensor<float, 3> u_seq(1, 10, wrapper.get_n_u());
  u_seq.fill(0.0);

  auto opt_rollout = sds::rollout_gpu(wrapper, integrator, x0, u_seq, 0.05f);
  cudaDeviceSynchronize();
  for (int t = 0; t < opt_rollout.shape(1); ++t)
  {
    std::cout << "t = " << t * 0.05f << "s, x = [";
    for (int i = 0; i < opt_rollout.shape(2); ++i)
      std::cout << opt_rollout(0, t, i) << ",";
    std::cout << "]\n";
  }

  return 0;
}