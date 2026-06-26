#include <iostream>

#include "sds_control.cuh"
#include "sds_core.cuh"
// #include <Eigen/Dense>
#include "tensor.cuh"

template <typename Scalar>
class DubinsCar
{
 public:
  using ScalarType = Scalar;

  __device__ __host__ void dynamics(
      Scalar t, const Scalar* x, const Scalar* u, Scalar* dxdt)
  {
    Scalar v = u[0];
    Scalar omega = u[1];
    Scalar theta = x[2];
    dxdt[0] = v * cos(theta);
    dxdt[1] = v * sin(theta);
    dxdt[2] = omega;
  }

  __host__ __device__ int get_n_x() const { return 3; }

  __host__ __device__ int get_n_u() const { return 2; }

  __host__ std::vector<std::string> get_x_names() const
  {
    return {"x", "y", "theta"};
  }

  __host__ std::vector<std::string> get_u_names() const
  {
    return {"v", "omega"};
  }

  __host__ __device__ Scalar* get_x_lower_bounds() const
  {
    static Scalar lb[3] = {-1e50, -1e50, -1e50};
    return lb;
  }

  __host__ __device__ Scalar* get_x_upper_bounds() const
  {
    static Scalar ub[3] = {1e50, 1e50, 1e50};
    return ub;
  }

  __host__ __device__ Scalar* get_u_lower_bounds() const
  {
    static Scalar lb[2] = {0.0f, -0.8f};
    return lb;
  }

  __host__ __device__ Scalar* get_u_upper_bounds() const
  {
    static Scalar ub[2] = {5.0f, 0.8f};
    return ub;
  }
};

__device__ float target_cost(
    TensorView<float, 1> x_target, TensorView<float, 2> x_seq,
    TensorView<float, 2> u_seq)
{
  // final quadratic cost to target
  float cost = 0.0f;
  for (int i = 0; i < x_seq.shape(1); ++i)
  {
    float diff = x_seq(x_seq.shape(0) - 1, i) - x_target(i);
    cost += diff * diff;
  }
  // running control cost
  for (int t = 0; t < u_seq.shape(0); ++t)
  {
    for (int i = 0; i < u_seq.shape(1); ++i)
    {
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

  // auto u_seq = sds::random_tensor<float>(10, 20, 2);
  //  u_seq.fill(0.1);

  // auto opt_rollout = sds::rollout_gpu(dubins_car, integrator, x0, u_seq,
  // 0.01f);
  // auto opt_rollout = sds::rollout_cpu(dubins_car, integrator, x0, u_seq,
  // 0.01f);

  //  std::cout << "b,t,x,y,theta" << std::endl;
  //  for (int b = 0; b < opt_rollout.shape(0); ++b)
  //    for (int i = 0; i < opt_rollout.shape(1); ++i)
  //      std::cout << b << "," << i << "," << opt_rollout(b, i, 0) << ","
  //                << opt_rollout(b, i, 1) << "," << opt_rollout(b, i, 2) <<
  //                "\n";
  //
  return 0;
  /*

    Tensor<float, 1> x_target(3);
    x_target(0) = 5.0f;
    x_target(1) = -5.0f;
    x_target(2) = 0.0f;
    auto cost = [target = x_target.view()] __device__(
                    TensorView<float, 2> x_seq,
                    TensorView<float, 2> u_seq) -> float
    { return target_cost(target, x_seq, u_seq); };

    // Initial mean control sequence [1, T, n_u], warm-started at small forward
    // velocity
    const int T = 20;
    float dt = 0.1f;
    Tensor<float, 3> u_mean(1, T, dubins_car.get_n_u());
    u_mean.fill(0.0f);

    // --- CEM optimisation -------------------------------------------------
    sds::cem(dubins_car, integrator, x0, u_mean, cost, dt, 1024, 100,
    250, 2.5f);

    // --- Final rollout on optimised mean ----------------------------------
    // Expand mean [1, T, n_u] — rollout_gpu handles batch dim naturally
    auto opt_rollout = sds::rollout_gpu(dubins_car, integrator, x0, u_mean, dt);
    cudaDeviceSynchronize();

    // --- CSV output -------------------------------------------------------
    std::cout << "b,t,x,y,theta" << std::endl;
    for (int b = 0; b < opt_rollout.shape(0); ++b)
      for (int i = 0; i < opt_rollout.shape(1); ++i)
        std::cout << b << "," << i << "," << opt_rollout(b, i, 0) << ","
                  << opt_rollout(b, i, 1) << "," << opt_rollout(b, i, 2) <<
    "\n";

    return 0;
  */
}