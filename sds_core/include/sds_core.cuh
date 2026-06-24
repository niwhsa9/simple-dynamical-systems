#include <concepts>
#include <iostream>
#include <memory>
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

template <typename I, typename Sys>
concept Integrator =
    DynamicalSystem<Sys> &&
    requires(
        I integrator, Sys sys, typename Sys::ScalarType dt,
        typename Sys::ScalarType t, typename Sys::ScalarType* x,
        typename Sys::ScalarType* u, typename Sys::ScalarType* x_next) {
      { integrator(sys, dt, t, x, u, x_next) };
    };

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

template <DynamicalSystem Sys, typename Integrator>
__global__ void rollout_kernel(
    Sys sys, Integrator integrator, const TensorView<const float, 1> x0,
    const TensorView<const float, 3> u_seq, float dt,
    TensorView<float, 3> x_seq)
{
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_size = u_seq.shape(0);
  if (batch_idx >= batch_size) return;
  int T = u_seq.shape(1);
  printf("[rollout_kernel] batch_idx = %d, T = %d\n", batch_idx, T);
  x_seq.slice_1d<2>(batch_idx, 0).deep_copy_from(x0);
  float t = 0.0;
  for (int i = 0; i < T; ++i)
  {
    integrator(
        sys, dt, t, x_seq.slice_1d<2>(batch_idx, i),
        u_seq.slice_1d<2>(batch_idx, i), x_seq.slice_1d<2>(batch_idx, i + 1));
    t += dt;
  }
}

template <DynamicalSystem Sys, typename Integrator>
Tensor<typename Sys::ScalarType, 3> rollout_gpu(
    const Sys& sys, const Integrator& integrator, const Tensor<float, 1>& x0,
    const Tensor<float, 3>& u_seq, float dt)
{
  int batch_size = u_seq.shape(0);
  int T = u_seq.shape(1);
  int n_x = sys.get_n_x();
  Tensor<float, 3> x_seq(batch_size, T + 1, n_x);
  rollout_kernel<<<(batch_size + 255) / 256, 256>>>(
      sys, integrator, x0.view(), u_seq.view(), dt, x_seq.view());
  return x_seq;
}

// One thread per rollout — evaluates CostFunc on that rollout's x_seq/u_seq
// slices and writes the scalar result into costs[b].
template <typename Cost>
__global__ void cost_kernel(
    Cost cost, TensorView<float, 3> x_seq,  // [N, T+1, n_x]
    TensorView<float, 3> u_seq,             // [N, T,   n_u]
    TensorView<float, 1> costs)             // [N]
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= x_seq.shape(0)) return;
  // 2-D views for this rollout — fix batch dim (0) at b.
  auto x_b = x_seq.slice<0>(b);  // [T+1, n_x]
  auto u_b = u_seq.slice<0>(b);  // [T,   n_u]
  costs(b) = cost(x_b, u_b);
}

template <DynamicalSystem Sys, typename Integ, typename Cost>
void cem(
    const Sys& sys, const Integ& integrator, const Tensor<float, 1>& x0,
    Tensor<float, 3>& u_seq,  // [1, T, n_u] — mean, written back
    Cost cost, float dt, int n_samples = 512, int n_elites = 64,
    int n_iters = 10, float sigma_init = 0.5f)
{
  const int T = u_seq.shape(1);
  const int n_u = u_seq.shape(2);

  // -- Distribution parameters (host-side, [T, n_u]) --------------------
  // Mean: copy from incoming u_seq
  std::vector<float> mean(T * n_u);
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < n_u; ++j) mean[t * n_u + j] = u_seq(0, t, j);

  std::vector<float> sigma(T * n_u, sigma_init);

  // -- Allocate persistent buffers ---------------------------------------
  Tensor<float, 3> samples(n_samples, T, n_u);  // [N, T, n_u]
  Tensor<float, 1> costs(n_samples);            // [N]

  std::mt19937 rng(42);
  std::normal_distribution<float> gauss(0.f, 1.f);

  const int threads = 256;
  const int blocks = (n_samples + threads - 1) / threads;

  for (int iter = 0; iter < n_iters; ++iter)
  {
    // -- Sample N control sequences from current diagonal Gaussian -----
    for (int n = 0; n < n_samples; ++n)
      for (int t = 0; t < T; ++t)
        for (int j = 0; j < n_u; ++j)
          samples(n, t, j) =
              mean[t * n_u + j] + sigma[t * n_u + j] * gauss(rng);

    // -- Rollout all samples on GPU ------------------------------------
    Tensor<float, 3> x_seq = rollout_gpu(sys, integrator, x0, samples, dt);

    // -- Evaluate costs on GPU, one thread per rollout -----------------
    cost_kernel<<<blocks, threads>>>(
        cost, x_seq.view(), samples.view(), costs.view());
    cudaDeviceSynchronize();

    // -- Sort by cost on host (unified memory — no memcpy needed) ------
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + n_elites, indices.end(),
        [&](int a, int b) { return costs(a) < costs(b); });

    // -- Refit mean and variance from elite set ------------------------
    std::fill(mean.begin(), mean.end(), 0.f);
    for (int k = 0; k < n_elites; ++k)
    {
      int n = indices[k];
      for (int t = 0; t < T; ++t)
        for (int j = 0; j < n_u; ++j) mean[t * n_u + j] += samples(n, t, j);
    }
    for (auto& v : mean) v /= float(n_elites);

    std::fill(sigma.begin(), sigma.end(), 0.f);
    for (int k = 0; k < n_elites; ++k)
    {
      int n = indices[k];
      for (int t = 0; t < T; ++t)
        for (int j = 0; j < n_u; ++j)
        {
          float diff = samples(n, t, j) - mean[t * n_u + j];
          sigma[t * n_u + j] += diff * diff;
        }
    }
    for (auto& v : sigma) v = std::sqrt(v / float(n_elites) + 1e-6f);
  }

  // -- Write optimized mean back into u_seq[0, :, :] --------------------
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < n_u; ++j) u_seq(0, t, j) = mean[t * n_u + j];
}

}  // namespace sds
