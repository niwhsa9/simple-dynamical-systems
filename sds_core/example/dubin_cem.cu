#include <iostream>
#include "sds_core.cuh"
//#include <Eigen/Dense>
#include "tensor.cuh"

template <typename Scalar>
class DubinsCar {

  public:
  using ScalarType = Scalar;
  __device__ __host__ void dynamics(Scalar t, const Scalar* x, const Scalar* u, Scalar* dxdt) {
    Scalar v = fmaxf(fminf(u[0], 5.0f), 5.0f); 
    Scalar omega = fmaxf(fminf(u[1], 0.8f), -0.8f); 
    Scalar theta = x[2];
    dxdt[0] = v * cos(theta);
    dxdt[1] = v * sin(theta);
    dxdt[2] = omega;
  }

  __host__ __device__ int get_n_x() const { return 3; }
  __host__ __device__ int get_n_u() const { return 2; }

};


__device__ float target_cost(TensorView<float, 1> x_target, TensorView<float, 2> x_seq, TensorView<float, 2> u_seq) {
  // final quadratic cost to target
  float cost = 0.0f;
  for (int i = 0; i < x_seq.shape(1); ++i) {
    float diff = x_seq(x_seq.shape(0) - 1, i) - x_target(i);
    cost += diff * diff;
  }
  // running control cost
  for (int t = 0; t < u_seq.shape(0); ++t) {
    for (int i = 0; i < u_seq.shape(1); ++i) {
      float u = u_seq(t, i);
      cost += 0.01f * u * u;
    }
  }
  return cost;
}

// device lambda for AV

int main()
{
  DubinsCar<float> dubins_car;
  sds::RK2 integrator;

  Tensor<float, 1> x0(3);
  x0.fill(0.0);
  
  Tensor<float, 3> u_seq(10, 20, 2);
  u_seq.fill(0.1);
  //auto opt_rollout = sds::rollout_gpu(dubins_car, integrator, x0, u_seq, 0.1f);
  cudaDeviceSynchronize();

  Tensor<float, 1> x_target(3);
  x_target(0) = 5.0f;
  x_target(1) = -5.0f;
  x_target(2) = 0.0f;
  auto cost = [target = x_target.view()]__device__(TensorView<float, 2> x_seq, TensorView<float, 2> u_seq) -> float{
    return target_cost(target, x_seq, u_seq);
  };


  // Initial mean control sequence [1, T, n_u], warm-started at small forward velocity
  const int T   = 20;
  float dt = 0.1f;
  Tensor<float, 3> u_mean(1, T, dubins_car.get_n_u());
  u_mean.fill(0.0f);

  // --- CEM optimisation -------------------------------------------------
  sds::cem(dubins_car, integrator, x0, u_mean, cost, dt, 1024, 100, 250, 2.5f);

  // --- Final rollout on optimised mean ----------------------------------
  // Expand mean [1, T, n_u] — rollout_gpu handles batch dim naturally
  auto opt_rollout = sds::rollout_gpu(dubins_car, integrator, x0, u_mean, dt);
  cudaDeviceSynchronize();

  // --- CSV output -------------------------------------------------------
  std::cout << "b,t,x,y,theta" << std::endl;
  for (int b = 0; b < opt_rollout.shape(0); ++b)
      for (int i = 0; i < opt_rollout.shape(1); ++i)
          std::cout << b << ","
                    << i << ","
                    << opt_rollout(b, i, 0) << ","
                    << opt_rollout(b, i, 1) << ","
                    << opt_rollout(b, i, 2) << "\n";

  

  return 0;
}