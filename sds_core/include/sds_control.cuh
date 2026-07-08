#pragma once
// #include <Eigen/Dense>
// #include <Eigen/Geometry>
#include <concepts>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include "sds_core.cuh"
#include "tensor.cuh"

namespace sds
{

// concept Cost = ... TODO(amg)

template <typename T, typename... Dims>
Tensor<T, sizeof...(Dims)> random_tensor(Dims... dims)
{
  Tensor<T, sizeof...(Dims)> tensor(dims...);
  std::mt19937 rng(42);
  std::normal_distribution<T> gauss(0.f, 1.f);
  for (int i = 0; i < tensor.numel(); ++i) tensor.data()[i] = gauss(rng);
  return tensor;
}

template <typename T, int dim>
Tensor<T, dim + 1> random_batch_tensor(
    const TensorView<T, dim>& mean, const TensorView<T, dim>& std_dev,
    int batch_size)
{
  // Tensor<T, dim + 1> tensor(batch_size, mean.shape());
  std::array<int, dim + 1> shape;
  shape[0] = batch_size;
  std::ranges::copy(std::span{mean.shape_}, shape.begin() + 1);
  Tensor<T, dim + 1> tensor(shape);

  std::mt19937 rng(42);
  for (int b = 0; b < batch_size; ++b)
    for (int i = 0; i < mean.numel(); ++i)
    {
      std::normal_distribution<T> gauss(mean.data()[i], std_dev.data()[i]);
      tensor.template slice<0>(b).data()[i] = gauss(rng);
    }
  return tensor;
}

__device__ inline float quadratic_target_cost(
    const TensorView<float, 1>& x_target, const TensorView<float, 1>& Qf,
    const TensorView<float, 1>& Rf, const TensorView<float, 2>& x_seq,
    const TensorView<float, 2>& u_seq)
{
  // final quadratic cost to target
  float cost = 0.0f;
  for (int i = 0; i < x_seq.shape(1); ++i)
  {
    float diff = x_seq(x_seq.shape(0) - 1, i) - x_target(i);
    cost += diff * diff * Qf(i);
  }
  // running control cost
  for (int t = 0; t < u_seq.shape(0); ++t)
  {
    for (int i = 0; i < u_seq.shape(1); ++i)
    {
      float u = u_seq(t, i);
      cost += Rf(i) * u * u;
    }
  }
  return cost;
}

//__device__ float quaternion_cost(
//    Eigen::Quaterniond<float, 1> q_target, const TensorView<float, 2> x_seq)
//{
//  //
//}

template <typename T>
std::tuple<Tensor<T, 2>, Tensor<T, 2>> cem_mean_std(
    const TensorView<T, 3>& u, const TensorView<T, 1>& costs, int num_elites)
{
  int batch_size = u.shape(0);
  int horizon = u.shape(1);
  int n_u = u.shape(2);

  Tensor<T, 2> mean(horizon, n_u);
  Tensor<T, 2> std_dev(horizon, n_u);

  std::vector<int> indices(batch_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(
      indices.begin(), indices.begin() + num_elites, indices.end(),
      [&](int a, int b) { return costs(a) < costs(b); });

  for (int t = 0; t < horizon; ++t)
    for (int j = 0; j < n_u; ++j)
    {
      mean(t, j) = 0.0;
      for (int k = 0; k < num_elites; ++k) mean(t, j) += u(indices[k], t, j);
      mean(t, j) /= float(num_elites);
    }

  for (int t = 0; t < horizon; ++t)
    for (int j = 0; j < n_u; ++j)
    {
      std_dev(t, j) = 0.0;
      for (int k = 0; k < num_elites; ++k)
      {
        float diff = u(indices[k], t, j) - mean(t, j);
        std_dev(t, j) += diff * diff;
      }
      std_dev(t, j) = std::sqrt(std_dev(t, j) / float(num_elites) + 1e-6f);
    }

  return std::make_tuple(std::move(mean), std::move(std_dev));
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

template <RolloutProvider<float> P, typename Cost>
Tensor<float, 2> cem(
    P& plant, const TensorView<float, 1>& x0, const Cost& cost, int T, int n_u,
    float dt, int n_samples = 512, int n_elites = 64, int n_iters = 100,
    float sigma_init = 0.5f)
{
  Tensor<float, 2> u_mean(T, n_u);
  u_mean.fill(0.0f);

  Tensor<float, 2> u_std_dev(T, n_u);
  u_std_dev.fill(sigma_init);

  for (int iter = 0; iter < n_iters; ++iter)
  {
    auto u_samples =
        random_batch_tensor(u_mean.view(), u_std_dev.view(), n_samples);
    auto x_samples = plant(x0, u_samples.view(), dt);  // [N, T+1, n_x]

    Tensor<float, 1> costs(n_samples);
    cost_kernel<<<(n_samples + 255) / 256, 256>>>(
        cost, x_samples.view(), u_samples.view(), costs.view());
    cudaDeviceSynchronize();

    std::tie(u_mean, u_std_dev) =
        cem_mean_std(u_samples.view(), costs.view(), n_elites);
  }
  return u_mean;
}

}  // namespace sds
