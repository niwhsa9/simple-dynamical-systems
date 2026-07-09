#pragma once
#include <Eigen/Dense>

#include "sds_core.cuh"
#include "tensor.cuh"

namespace sds
{

// zero order hold re-roll at finer dt
// template <typename RolloutProvider<float>> f;
// Tensor<float, 2> zoh_reroll(Tensor<float, 2> u, );
// todo
// keep_tape_after_time
Tensor<float, 2> keep_tape_after_time(
    const TensorView<float, 2>& x_seq, double start_time, double t, double dt,
    int pad_to_T);

// TODO(amg) probably worth thinking about how this thing ends up on GPU,
// LinearPolicy a nd LinearPolicyView? Yuck.
struct LinearPolicy
{
  Tensor<float, 3> K;          // N x nx x nu row major
  Tensor<float, 2> x_nominal;  // N x nx row major
  Tensor<float, 2> u_nominal;  // N x nu row major
  double start_time, dt;
  int nX, nU;

  LinearPolicy(
      Tensor<float, 3>&& K_, Tensor<float, 2>&& x_nominal_,
      Tensor<float, 2>&& u_nominal_, double start_time_, double dt_)
      : K(std::move(K_)), x_nominal(std::move(x_nominal_)),
        u_nominal(std::move(u_nominal_)), start_time(start_time_), dt(dt_),
        nX(x_nominal.shape(1)), nU(u_nominal.shape(1))
  {
  }

  LinearPolicy(const LinearPolicy& other)
      : K(other.K.clone()), x_nominal(other.x_nominal.clone()),
        u_nominal(other.u_nominal.clone()), start_time(other.start_time),
        dt(other.dt), nX(other.nX), nU(other.nU)
  {
  }

  // copy assignment
  LinearPolicy& operator=(const LinearPolicy& other)
  {
    if (this != &other)
    {
      K = other.K.clone();
      x_nominal = other.x_nominal.clone();
      u_nominal = other.u_nominal.clone();

      start_time = other.start_time;
      dt = other.dt;
      nX = other.nX;
      nU = other.nU;
    }
    return *this;
  }

  Tensor<float, 1> eval_policy(double t, const float* x) const
  {
    // ZOH: clamp to step_lower
    int step = static_cast<int>(std::floor((t - start_time) / dt));
    if (step < 0 || step >= x_nominal.shape(0))
      throw std::runtime_error("Time t is out of bounds for policy evaluation");

    // u = u_nominal[step] + K[step] * (x - x_nominal[step])
    Eigen::Map<const Eigen::VectorXf> x_eigen(x, nX);
    Eigen::Map<const Eigen::VectorXf> x_nom(
        x_nominal.slice_1d<1>(step).data(), nX);
    Eigen::Map<const Eigen::VectorXf> u_nom(
        u_nominal.slice_1d<1>(step).data(), nU);
    Eigen::Map<const Eigen::MatrixXf> K_step(K.slice<0>(step).data(), nU, nX);

    Eigen::VectorXf u = u_nom + K_step * (x_eigen - x_nom);

    Tensor<float, 1> result(nU);
    std::memcpy(result.data(), u.data(), nU * sizeof(float));
    return result;
  }
};

std::pair<Tensor<float, 2>, Tensor<float, 2>> simulate_plant_with_policy(
    RolloutProvider<float> auto& plant, const LinearPolicy& policy,
    const TensorView<float, 1>& x0, double dt, double t, int T)

{
  Tensor<float, 2> x_traj(T + 1, policy.nX);
  Tensor<float, 2> u_traj(T, policy.nU);

  // Copy x0 into x_traj[0, :]
  x_traj.slice_1d<1>(0).deep_copy_from(x0);

  for (int i = 0; i < T; ++i)
  {
    double t_i = t + i * dt;

    // Evaluate policy at current state
    auto u = policy.eval_policy(t_i, x_traj.slice_1d<1>(i).data());

    // Write u into u_traj[i, :]
    u_traj.slice_1d<1>(i).deep_copy_from(u.view());

    // Pack u into a [1, 1, nU] Tensor for the RolloutProvider
    Tensor<float, 3> u_seq(1, 1, policy.nU);
    for (int j = 0; j < policy.nU; ++j) u_seq(0, 0, j) = u(j);

    // Plant takes x_current as [nX] view, u_seq as [1, 1, nU]
    auto x_next_batch =
        plant(x_traj.slice_1d<1>(i), u_seq.view(), static_cast<float>(dt));

    // x_next_batch is [1, 2, nX] (batch=1, T+1=2 states), take x_next = [1, nX]
    x_traj.template slice_1d<1>(i + 1).deep_copy_from(
        x_next_batch.template slice_1d<2>(0, 1));
  }

  return {std::move(x_traj), std::move(u_traj)};
}

// Computes TVLQR gains, Tensor in/out.
// A: N x nX x nX row-major   (dynamics jacobian wrt x, per step)
// B: N x nX x nU row-major   (dynamics jacobian wrt u, per step)
// Q, Qf: nX x nX row-major
// R: nU x nU row-major
// Returns K: N x nX x nU, laid out so that (per the LinearPolicy convention)
// K.slice<0>(i) reinterpreted col-major as (nU, nX) gives the actual gain
// matrix u = u_nom + K_i * (x - x_nom).
Tensor<float, 3> compute_tvlqr_gains(
    const TensorView<float, 3>& A, const TensorView<float, 3>& B,
    const TensorView<float, 2>& Q, const TensorView<float, 2>& Qf,
    const TensorView<float, 2>& R, float dt, bool discretize = true);

template <DynamicalSystem Dyn>
std::pair<Tensor<float, 3>, Tensor<float, 3>> get_linearized_trajectory(
    const Dyn& system, const TensorView<float, 2>& x_traj,
    const TensorView<float, 2>& u_traj, float dt)
{
  const int N = x_traj.shape(0);
  const int nX = system.get_n_x();
  const int nU = system.get_n_u();

  Tensor<float, 3> A(N, nX, nX);
  Tensor<float, 3> B(N, nX, nU);

  float t = 0.0f;
  for (int i = 0; i < N; ++i)
  {
    TensorView<float, 1> x_i = x_traj.slice_1d<1>(i);
    TensorView<float, 1> u_i = u_traj.slice_1d<1>(i);
    TensorView<float, 2> A_i = A.slice<0>(i);
    TensorView<float, 2> B_i = B.slice<0>(i);

    system.get_dfdx(t, x_i.data(), u_i.data(), A_i.data());
    system.get_dfdu(t, x_i.data(), u_i.data(), B_i.data());
    t += dt;
  }

  return {std::move(A), std::move(B)};
}

}  // namespace sds