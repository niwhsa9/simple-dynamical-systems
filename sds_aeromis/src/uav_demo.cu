#include <aeromis_core/dynamical_systems/aerial_vehicle.hpp>
#include <filesystem>
#include <iostream>

#include "aeromis_core/util/array.hpp"
#include "sds_core.cuh"
#include "sds_control.cuh"
#include "sds_aeromis/aerial_vehicle.cuh"
using namespace sds_aeromis;
int main()
{
  AerialVehicleManager manager(std::filesystem::path(
      "/home/ashwin/sources/aeromis/src/aeromis_simulator/urdf/edge540_24/"
      "edge540_24.uardf"));
  sds::RK2 integrator;

  Tensor<float, 1> x0(manager.get_aerial_vehicle_gpu().get_n_x());
  x0(3) = 1.0;
  x0(7) = 7.0;

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

  auto cost =
      [target = x_target.view(), Qf = Qf.view(), R = R.view()] __device__(
          const TensorView<float, 2> &x_seq, const TensorView<float, 2> &u_seq) -> float
  { return sds::quadratic_target_cost(target, Qf, R, x_seq, u_seq); };

  float dt = 0.05f;
  auto tape = sds::cem(
      manager.get_aerial_vehicle_gpu(), integrator, x0, cost, 20, dt, 1024, 100,
      180, 0.5f);

  TensorView<float, 3> tape_view = tape.view().unsqueeze();
  auto opt_rollout = sds::rollout_gpu(
      manager.get_aerial_vehicle_gpu(), integrator, x0.view(), tape_view, dt);

  rollout_to_csv(manager, opt_rollout.template slice<0>(0), tape.view(), dt);


  return 0;
}