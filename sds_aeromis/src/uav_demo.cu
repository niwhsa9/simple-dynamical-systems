#include <aeromis_core/dynamical_systems/aerial_vehicle.hpp>
#include <filesystem>
#include <iostream>

#include "aeromis_core/util/array.hpp"
#include "sds_core.cuh"
#include "sds_control.cuh"

template <typename Mem, typename Scalar>
class AerialVehicle
{
 public:
  using ScalarType = Scalar;

  __host__ AerialVehicle(
      aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle,
      aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle_cpu,
      uint8_t* memory_buffer)
      : aerial_vehicle_(aerial_vehicle),
        aerial_vehicle_cpu_(aerial_vehicle_cpu), memory_buffer(memory_buffer)
  {
    n_x = aerial_vehicle_cpu->get_n_x();
    n_u = aerial_vehicle_cpu->get_n_u();
  }

  __device__ __host__ void dynamics(
      Scalar t, const Scalar* x, const Scalar* u, Scalar* dxdt)
  {
#ifdef __CUDA_ARCH__
    int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
#else
    int batch_idx = 0;
#endif
    aeromis_core::Array<Mem, Scalar> x_view(const_cast<Scalar*>(x), n_x);
    aeromis_core::Array<Mem, Scalar> u_view(const_cast<Scalar*>(u), n_u);
    aeromis_core::Array<Mem, Scalar> dxdt_view(dxdt, n_x);
    aerial_vehicle_->dynamics(
        t, x_view, u_view, dxdt_view, batch_idx,
        memory_buffer +
            batch_idx * aerial_vehicle_->required_bytes_per_thread());
  }

  __host__ __device__ int get_n_x() const { return n_x; }

  __host__ __device__ int get_n_u() const { return n_u; }

  __host__ std::vector<std::string> get_x_names() const
  {
    return aerial_vehicle_cpu_->get_state_parameters().get_names();
  }

  __host__ std::vector<std::string> get_u_names() const
  {
    return aerial_vehicle_cpu_->get_input_parameters().get_names();
  }

  __host__ __device__ Scalar* get_x_upper_bounds() const
  {
#ifdef __CUDA_ARCH__
    auto* ptr = aerial_vehicle_;
#else
    auto* ptr = aerial_vehicle_cpu_;
#endif
    return ptr->get_state_parameters_ref().get_upper_bounds_ref().get_data();
  }

  __host__ __device__ Scalar* get_x_lower_bounds() const
  {
#ifdef __CUDA_ARCH__
    auto* ptr = aerial_vehicle_;
#else
    auto* ptr = aerial_vehicle_cpu_;
#endif
    return ptr->get_state_parameters_ref().get_lower_bounds_ref().get_data();
  }

  __host__ __device__ Scalar* get_u_upper_bounds() const
  {
#ifdef __CUDA_ARCH__
    auto* ptr = aerial_vehicle_;
#else
    auto* ptr = aerial_vehicle_cpu_;
#endif
    return ptr->get_input_parameters_ref().get_upper_bounds_ref().get_data();
  }

  __host__ __device__ Scalar* get_u_lower_bounds() const
  {
#ifdef __CUDA_ARCH__
    auto* ptr = aerial_vehicle_;
#else
    auto* ptr = aerial_vehicle_cpu_;
#endif
    return ptr->get_input_parameters_ref().get_lower_bounds_ref().get_data();
  }

 private:
  aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle_;
  aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle_cpu_;
  uint8_t* memory_buffer;
  int n_x, n_u;
};

class AerialVehicleManager
{
 public:
  __host__ AerialVehicleManager(const std::filesystem::path& description_path)
  {
    aeromis_core::AerialVehicleDescription<aeromis_core::GPU, float>
        description(description_path);

    aeromis_core::AerialVehicleDescription<aeromis_core::CPU, double>
        description_cpu(description_path);

    aerial_vehicle_gpu_ =
        std::make_unique<aeromis_core::AerialVehicle<aeromis_core::GPU, float>>(
            std::move(description));
    aerial_vehicle_gpu_->copy_members_to_gpu(
        aerial_vehicle_gpu_->get_gpu_instance());
    cudaMallocManaged(
        &gpu_memory_buffer,
        aerial_vehicle_gpu_->required_bytes_per_thread() * 1024);

    aerial_vehicle_cpu_ = std::make_unique<
        aeromis_core::AerialVehicle<aeromis_core::CPU, double>>(
        std::move(description_cpu));
    cpu_memory_buffer.resize(
        aerial_vehicle_cpu_->required_bytes_per_thread() * 1024);
  }

  __device__ __host__ ~AerialVehicleManager() { cudaFree(gpu_memory_buffer); }

  __host__ auto get_aerial_vehicle_gpu()
  {
    return AerialVehicle<aeromis_core::GPU, float>(
        aerial_vehicle_gpu_->get_gpu_instance(), aerial_vehicle_gpu_.get(),
        gpu_memory_buffer);
  }

  __host__ auto get_aerial_vehicle_cpu()
  {
    return AerialVehicle<aeromis_core::CPU, double>(
        aerial_vehicle_cpu_.get(), aerial_vehicle_cpu_.get(),
        cpu_memory_buffer.data());
  }

 private:
  std::unique_ptr<aeromis_core::AerialVehicle<aeromis_core::CPU, double>>
      aerial_vehicle_cpu_;
  std::unique_ptr<aeromis_core::AerialVehicle<aeromis_core::GPU, float>>
      aerial_vehicle_gpu_;
  uint8_t* gpu_memory_buffer;
  std::vector<uint8_t> cpu_memory_buffer;
};

void rollout_to_csv(
    AerialVehicleManager& manager, const TensorView<float, 2>& x_seq,
    const TensorView<float, 2>& u_seq, float dt)
{
  // csv header
  for (auto name : manager.get_aerial_vehicle_gpu().get_x_names())
    std::cout << name << ",";

  for (auto name : manager.get_aerial_vehicle_gpu().get_u_names())
    std::cout << name << ",";

  std::cout << std::endl;

  for (int t = 0; t < x_seq.shape(0); ++t)
  {
    for (int i = 0; i < x_seq.shape(1); ++i) std::cout << x_seq(t, i) << ",";
    for (int i = 0; i < u_seq.shape(1); ++i)
      std::cout << u_seq(std::min(t, u_seq.shape(0) - 1), i) << ",";
    std::cout << std::endl;
  }
}

int main()
{
  AerialVehicleManager manager(std::filesystem::path(
      "/home/ashwin/sources/aeromis/src/aeromis_simulator/urdf/edge540_24/"
      "edge540_24.uardf"));
  sds::RK2 integrator;

  Tensor<float, 1> x0(manager.get_aerial_vehicle_gpu().get_n_x());
  x0(3) = 1.0;
  x0(7) = 7.0;

  // Tensor<float, 3> u_seq(1, 10, manager.get_aerial_vehicle_gpu().get_n_u());
  // u_seq.fill(11);
  // auto opt_rollout = sds::rollout_gpu(
  //     manager.get_aerial_vehicle_gpu(), integrator, x0, u_seq, 0.05f);
  // csv header
  //rollout_to_csv(manager, opt_rollout.slice<0>(0), u_seq.slice<0>(0), dt);

  Tensor<float, 1> x_target(manager.get_aerial_vehicle_gpu().get_n_x());
  x_target.fill(0.0f);
  x_target(0) = 3.5f;
  x_target(3) = 0.923;
  x_target(4) = 0.0;
  x_target(5) = -0.3826834;
  x_target(6) = 0.0;
  x_target(7) = 0.5;
  x_target(9) = -0.5;

  Tensor<float, 1> Qf(manager.get_aerial_vehicle_gpu().get_n_x());
  Qf.fill(0.1f);
  Qf(0) = 400;
  Qf(1) = 400;
  Qf(2) = 400;

  Qf(3) = 10;
  Qf(4) = 10;
  Qf(5) = 10;
  Qf(6) = 10;

  Qf(7) = 1.0;
  Qf(8) = 1.0;
  Qf(9) = 1.0;
  Tensor<float, 1> R(manager.get_aerial_vehicle_gpu().get_n_u());
  R.fill(0.01f);
  R(3) = 10000;

  auto cost =
      [target = x_target.view(), Qf = Qf.view(), R = R.view()] __device__(
          const TensorView<float, 2> &x_seq, const TensorView<float, 2> &u_seq) -> float
  { return sds::quadratic_target_cost(target, Qf, R, x_seq, u_seq); };

  float dt = 0.05f;
  auto tape = sds::cem(
      manager.get_aerial_vehicle_gpu(), integrator, x0, cost, 20, dt, 1024, 100,
      200, 0.5f);

  TensorView<float, 3> tape_view = tape.view().unsqueeze();
  auto opt_rollout = sds::rollout_gpu(
      manager.get_aerial_vehicle_gpu(), integrator, x0.view(), tape_view, dt);

  rollout_to_csv(manager, opt_rollout.slice<0>(0), tape.view(), dt);


  return 0;
}