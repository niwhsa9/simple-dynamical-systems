#include <iostream>

#include "sds_control.cuh"
#include "sds_core.cuh"
#include "sds_tvlqr.cuh"
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

  __device__ __host__ void get_dfdx(
      Scalar t, const Scalar* x, const Scalar* u, Scalar* dfdx)
  {
    dfdx[0] = 0.0f;
    dfdx[1] = 0.0f;
    dfdx[2] = -u[0] * sin(x[2]);

    dfdx[3] = 0.0f;
    dfdx[4] = 0.0f;
    dfdx[5] = u[0] * cos(x[2]);

    dfdx[6] = 0.0f;
    dfdx[7] = 0.0f;
    dfdx[8] = 0.0f;
  }

  __device__ __host__ void get_dfdu(
      Scalar t, const Scalar* x, const Scalar* u, Scalar* dfdu)
  {
    dfdu[0] = cos(x[2]);
    dfdu[1] = 0.0f;

    dfdu[2] = sin(x[2]);
    dfdu[3] = 0.0f;

    dfdu[4] = 0.0f;
    dfdu[5] = 1.0f;
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

void dubin_to_csv(bool do_header, const Tensor<float, 2>& traj, int batch_index)
{
  if (do_header) std::cout << "b,t,x,y,theta" << std::endl;
  for (int i = 0; i < traj.shape(0); ++i)
  {
    std::cout << batch_index << "," << i << "," << traj(i, 0) << ","
              << traj(i, 1) << "," << traj(i, 2) << "\n";
  }
}

// device lambda for AV

int main()
{
  // Create a Dubins car system
  DubinsCar<float> dubins_car;
  sds::RK2 integrator;
  Tensor<float, 1> x0(3);
  x0.fill(0.0);

  // Generate nominal trajectory
  Tensor<float, 1> x_target(dubins_car.get_n_x());
  x_target.fill(0.0f);
  x_target(0) = 5.0f;
  x_target(1) = 3.0f;
  x_target(2) = 0.1f;

  Tensor<float, 1> Qf(dubins_car.get_n_x());
  Qf.fill(0.1f);
  Qf(0) = 400;
  Qf(1) = 400;
  Qf(2) = 400;

  Tensor<float, 1> R(dubins_car.get_n_u());
  R.fill(0.01f);

  auto cost =
      [target = x_target.view(), Qf = Qf.view(), R = R.view()] __device__(
          const TensorView<float, 2>& x_seq,
          const TensorView<float, 2>& u_seq) -> float
  { return sds::quadratic_target_cost(target, Qf, R, x_seq, u_seq); };

  float dt = 0.05f;

  auto dubin_plant = [&](const TensorView<float, 1>& x0,
                         TensorView<float, 3> u_seq,
                         float dt) -> Tensor<float, 3>
  { return sds::rollout_gpu(dubins_car, integrator, x0, u_seq, dt); };

  int n_u = dubins_car.get_n_u();
  auto tape =
      sds::cem(dubin_plant, x0, cost, 40, n_u, dt, 1024, 100, 180, 0.5f);

  TensorView<float, 3> tape_view = tape.view().unsqueeze();
  auto nominal_traj_x =
      sds::rollout_gpu(dubins_car, integrator, x0.view(), tape_view, dt);

  // Generate local TVLQR feedback
  Tensor<float, 3> As(
      tape.shape(0), dubins_car.get_n_x(), dubins_car.get_n_x());
  Tensor<float, 3> Bs(
      tape.shape(0), dubins_car.get_n_x(), dubins_car.get_n_u());
  Tensor<float, 2> LQR_Qf(dubins_car.get_n_x(), dubins_car.get_n_x());
  Tensor<float, 2> LQR_R(dubins_car.get_n_u(), dubins_car.get_n_u());
  Tensor<float, 2> LQR_Q(dubins_car.get_n_x(), dubins_car.get_n_x());
  LQR_Qf.fill(0.0f);
  LQR_R.fill(0.0f);
  LQR_Q.fill(0.0f);
  for (int i = 0; i < dubins_car.get_n_x(); ++i) LQR_Qf(i, i) = Qf(i);
  for (int i = 0; i < dubins_car.get_n_u(); ++i) LQR_R(i, i) = R(i);
  for (int i = 0; i < tape.shape(0); ++i)
  {
    TensorView<float, 1> x_i = nominal_traj_x.slice_1d<2>(1, i);
    TensorView<float, 1> u_i = tape.slice_1d<1>(i);
    TensorView<float, 2> A_i = As.slice<0>(i);
    TensorView<float, 2> B_i = Bs.slice<0>(i);
    dubins_car.get_dfdx(0.0f, x_i.data(), u_i.data(), A_i.data());
    dubins_car.get_dfdu(0.0f, x_i.data(), u_i.data(), B_i.data());
  }
  auto Ks = sds::compute_tvlqr_gains(
      As, Bs, LQR_Q.view(), LQR_Qf.view(), LQR_R.view(), dt);

  auto nominal_traj_2d = nominal_traj_x.clone().squeeze();

  sds::LinearPolicy lqr_policy(
      std::move(Ks), std::move(nominal_traj_2d), std::move(tape), 0.0, dt);

  Tensor<float, 1> x_perturb(3);
  x_perturb.fill(0.0);
  x_perturb(0) = 0.2f;
  x_perturb(1) = -0.2f;

  auto [x_sim, u_sim] = sds::simulate_plant_with_policy(
      dubin_plant, lqr_policy, x_perturb.view(), dt, 0.0, tape.shape(0));

  // Log to CSV
  dubin_to_csv(true, nominal_traj_x.clone().squeeze(), 0);
  dubin_to_csv(false, x_sim, 1);

  return 0;
}