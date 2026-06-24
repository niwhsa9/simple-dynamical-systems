#include <aeromis_core/dynamical_systems/aerial_vehicle.hpp>
#include <filesystem>
#include <iostream>

#include "aeromis_core/util/array.hpp"
#include "sds_core.cuh"


template <typename Mem, typename Scalar>
class AerialVehicle
{
 public:
  using ScalarType = Scalar;

  __host__ AerialVehicle(aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle, 
    aeromis_core::AerialVehicle<Mem, Scalar>* aerial_vehicle_cpu,
    uint8_t* memory_buffer)
      : aerial_vehicle_(aerial_vehicle), aerial_vehicle_cpu_(aerial_vehicle_cpu), memory_buffer(memory_buffer)
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
    aeromis_core::Array<Mem, Scalar> x_view(
        const_cast<Scalar*>(x), n_x);
    aeromis_core::Array<Mem, Scalar> u_view(
        const_cast<Scalar*>(u), n_u);
    aeromis_core::Array<Mem, Scalar> dxdt_view(
        dxdt, n_x);
    printf("batch_idx = %d, t = %f, x = [", batch_idx, t);
    printf("x_view size = %d, u_view size = %d, dxdt_view size = %d\n", x_view.get_size(), u_view.get_size(), dxdt_view.get_size());
    printf("address of x_view = %p, address of u_view = %p, address of dxdt_view = %p\n", x_view.get_data(), u_view.get_data(), dxdt_view.get_data());
    printf("address of memory_buffer = %p\n", memory_buffer);
    aerial_vehicle_->dynamics(
        t, x_view, u_view, dxdt_view, batch_idx,
        memory_buffer + batch_idx * aerial_vehicle_->required_bytes_per_thread());  
    //printf("[dyn] t = %f, dxdt = [", t);
    //for (int i = 0; i < aerial_vehicle_->get_n_x(); ++i)
    //  printf("%f,", dxdt[i]);
  }

  __host__ __device__ int get_n_x() const { return n_x; }

  __host__ __device__ int get_n_u() const { return n_u; }


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

    aerial_vehicle_gpu_ = std::make_unique<aeromis_core::AerialVehicle<aeromis_core::GPU, float>>(
            std::move(description));
    aerial_vehicle_gpu_->copy_members_to_gpu(aerial_vehicle_gpu_->get_gpu_instance());
    cudaMallocManaged(
        &gpu_memory_buffer, aerial_vehicle_gpu_->required_bytes_per_thread()*1024);

    aerial_vehicle_cpu_ = std::make_unique<aeromis_core::AerialVehicle<aeromis_core::CPU, double>>(
            std::move(description_cpu));
    cpu_memory_buffer.resize(aerial_vehicle_cpu_->required_bytes_per_thread()*1024);
  }

  __device__ __host__ ~AerialVehicleManager()
  {
      cudaFree(gpu_memory_buffer);
  }
  __host__ auto get_aerial_vehicle_gpu() { return AerialVehicle<aeromis_core::GPU, float>(aerial_vehicle_gpu_->get_gpu_instance(), aerial_vehicle_gpu_.get(), gpu_memory_buffer); }
  __host__ auto get_aerial_vehicle_cpu() { return AerialVehicle<aeromis_core::CPU, double>(aerial_vehicle_cpu_.get(), aerial_vehicle_cpu_.get(), cpu_memory_buffer.data()); }
 private:
  std::unique_ptr<aeromis_core::AerialVehicle<aeromis_core::CPU, double>> aerial_vehicle_cpu_;
  std::unique_ptr<aeromis_core::AerialVehicle<aeromis_core::GPU, float>> aerial_vehicle_gpu_;
  uint8_t* gpu_memory_buffer;
  std::vector<uint8_t> cpu_memory_buffer;
};

int main()
{
  AerialVehicleManager manager(std::filesystem::path("/home/ashwin/sources/aeromis/src/aeromis_simulator/urdf/edge540_24/edge540_24.uardf"));
  sds::RK2 integrator;

  Tensor<float, 1> x0(manager.get_aerial_vehicle_gpu().get_n_x());
  x0(3) = 1.0;
  x0(7) = 7.0;

  Tensor<float, 3> u_seq(1, 10, manager.get_aerial_vehicle_gpu().get_n_u());
  u_seq.fill(0.0);

  //Tensor<double, 1> x0(manager.get_aerial_vehicle_gpu().get_n_x());
  //x0(3) = 1.0;
  //x0(7) = 7.0;
  //Tensor<double, 3> u_seq(1, 10, manager.get_aerial_vehicle_gpu().get_n_u());
  //u_seq.fill(0.0);
  //Tensor<double, 1> dxdt (manager.get_aerial_vehicle_gpu().get_n_x());

  //manager.get_aerial_vehicle_cpu().dynamics(0.0, x0.data(), u_seq.slice_1d<2>(0, 0).data(), dxdt.data());

  auto opt_rollout = sds::rollout_gpu(manager.get_aerial_vehicle_gpu(), integrator, x0, u_seq, 0.05f);
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