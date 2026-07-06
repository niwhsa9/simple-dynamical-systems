#pragma once
#include <concepts>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "tensor.cuh"

namespace sds
{

template <typename Sys>
concept DynamicalSystem =
    // std::floating_point<typename Sys::ScalarType> &&
    requires(
        Sys sys, typename Sys::ScalarType t, typename Sys::ScalarType* x,
        typename Sys::ScalarType* u, typename Sys::ScalarType* dxdt) {
      typename Sys::ScalarType;
      { sys.get_n_x() } -> std::convertible_to<int>;
      { sys.get_n_u() } -> std::convertible_to<int>;
      { sys.dynamics(t, x, u, dxdt) };
    };

template <typename Sys>
concept BoundedSystem = DynamicalSystem<Sys> && requires(Sys sys) {
  {
    sys.get_x_lower_bounds()
  } -> std::convertible_to<const typename Sys::ScalarType*>;
  {
    sys.get_x_upper_bounds()
  } -> std::convertible_to<const typename Sys::ScalarType*>;
  {
    sys.get_u_lower_bounds()
  } -> std::convertible_to<const typename Sys::ScalarType*>;
  {
    sys.get_u_upper_bounds()
  } -> std::convertible_to<const typename Sys::ScalarType*>;
};

template <typename I, typename Sys>
concept Integrator =
    DynamicalSystem<Sys> &&
    requires(
        I integrator, Sys sys, typename Sys::ScalarType dt,
        typename Sys::ScalarType t, typename Sys::ScalarType* x,
        typename Sys::ScalarType* u, typename Sys::ScalarType* x_next) {
      { integrator(sys, dt, t, x, u, x_next) };
    };

// A thing that does Rollouts given an x0 and inputs. Effectively a "Plant"
template <typename P, typename Scalar>
concept RolloutProvider = requires(
    P plant, TensorView<Scalar, 1> x0, TensorView<Scalar, 3> u_seq, Scalar dt) {
  { plant(x0, u_seq, dt) } -> std::convertible_to<Tensor<Scalar, 3>>;
};

// template <typename P>
// concept Policy = requires(P policy, float t, ) {
//   { policy(t, u) };
// };

// NOTE(amg): RK2 can allocate it's own memory as needed, but in fact in this
// case, does not even need to be a functor at all
class RK2
{
 public:
  template <DynamicalSystem Sys>
  __host__ __device__ void operator()(
      Sys& sys, typename Sys::ScalarType dt, typename Sys::ScalarType t,
      const typename Sys::ScalarType* x, const typename Sys::ScalarType* u,
      typename Sys::ScalarType* x_next)
  {
    typename Sys::ScalarType k1[64];
    typename Sys::ScalarType k2[64];

    sys.dynamics(t, x, u, k1);
    for (int i = 0; i < sys.get_n_x(); ++i) x_next[i] = x[i] + dt * k1[i];
    sys.dynamics(t, x_next, u, k2);
    for (int i = 0; i < sys.get_n_x(); ++i)
      x_next[i] = x[i] + dt * 0.5f * (k1[i] + k2[i]);
  }
};

template <typename Scalar>
__host__ __device__ void clamp_to_bounds(
    const Scalar* lb, const Scalar* ub, int n_u, Scalar* u)
{
  for (int i = 0; i < n_u; ++i) u[i] = fmaxf(fminf(u[i], ub[i]), lb[i]);
}

template <DynamicalSystem Sys, typename Integrator>
__host__ __device__ void rollout(
    int batch_idx, Sys& sys, Integrator& integrator,
    const TensorView<float, 1>& x0, TensorView<float, 3>& u_seq, float dt,
    TensorView<float, 3>& x_seq)
{
  int T = u_seq.shape(1);
  x_seq.slice_1d<2>(batch_idx, 0).deep_copy_from(x0);
  float t = 0.0;
  for (int i = 0; i < T; ++i)
  {
    if constexpr (BoundedSystem<Sys>)
    {
      clamp_to_bounds(
          sys.get_u_lower_bounds(), sys.get_u_upper_bounds(), sys.get_n_u(),
          u_seq.slice_1d<2>(batch_idx, i).data());
    }
    integrator(
        sys, dt, t, x_seq.slice_1d<2>(batch_idx, i),
        u_seq.slice_1d<2>(batch_idx, i), x_seq.slice_1d<2>(batch_idx, i + 1));

    if constexpr (BoundedSystem<Sys>)
    {
      clamp_to_bounds(
          sys.get_x_lower_bounds(), sys.get_x_upper_bounds(), sys.get_n_x(),
          x_seq.slice_1d<2>(batch_idx, i + 1).data());
    }
    t += dt;
  }
}

template <DynamicalSystem Sys, typename Integrator>
__global__ void rollout_kernel(
    Sys sys, Integrator integrator, const TensorView<float, 1> x0,
    TensorView<float, 3> u_seq, float dt, TensorView<float, 3> x_seq)
{
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_size = u_seq.shape(0);
  if (batch_idx >= batch_size) return;
  rollout(batch_idx, sys, integrator, x0, u_seq, dt, x_seq);
}

// Note u is non-const due to bounds clamping
// TODO(amg), maybe this thing need not to take a system at all, but rather a
// function pointer to integrate and a struct with n_x, n_u, and bounds.
template <DynamicalSystem Sys, typename Integrator>
Tensor<typename Sys::ScalarType, 3> rollout_gpu(
    const Sys& sys, const Integrator& integrator,
    const TensorView<float, 1>& x0, TensorView<float, 3>& u_seq, float dt,
    std::optional<std::tuple<int, int>> grid_block = std::nullopt)
{
  int batch_size = u_seq.shape(0);
  int T = u_seq.shape(1);
  int n_x = sys.get_n_x();
  Tensor<float, 3> x_seq(batch_size, T + 1, n_x);

  int grid = (batch_size + 255) / 256;
  int block = 256;
  if (grid_block.has_value())
  {
    grid = std::get<0>(grid_block.value());
    block = std::get<1>(grid_block.value());
  }
  rollout_kernel<<<grid, block>>>(sys, integrator, x0, u_seq, dt, x_seq.view());
  CUDA_CHECK(cudaDeviceSynchronize());
  return x_seq;
}

template <DynamicalSystem Sys, typename Integrator>
Tensor<typename Sys::ScalarType, 3> rollout_cpu(
    const Sys& sys, const Integrator& integrator,
    const TensorView<float, 1>& x0, TensorView<float, 3>& u_seq, float dt)
{
  int batch_size = u_seq.shape(0);
  int T = u_seq.shape(1);
  int n_x = sys.get_n_x();
  Tensor<float, 3> x_seq(batch_size, T + 1, n_x);
  for (int b = 0; b < batch_size; ++b)
    rollout(b, sys, integrator, x0, u_seq, dt, x_seq.view());
  return x_seq;
}

}  // namespace sds