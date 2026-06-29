#pragma once
#include <Eigen/Dense>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>

#include "sds_core.cuh"
#include "tensor.cuh"

namespace sds
{

// This thing should never-ever be copied to the GPU. Recommend emitting Views
// to each tensor
struct LinearPolicy
{
  Tensor<float, 3> K;
  Tensor<float, 2> x_nominal;
  Tensor<float, 2> u_nominal;
  float start_time;
  float dt;

  // TODO(amg) this can wrap a simpler __host__, __device__ function with raw
  // pointer interfaces
  Tensor<float, 1> eval_policy(double t, const float* x, const float* u)
  {
    // Convert x and u to eigen
    Eigen::Map<const Eigen::VectorXf> x_eigen(x, x_nominal.shape(1));
    Eigen::Map<const Eigen::VectorXf> u_eigen(u, u_nominal.shape(1));
    // Compute the index of the time step
    int step_lower = std::floor((t - start_time) / dt);
    int step_upper = step_lower + 1;
    float alpha = (t - start_time) / dt - step_lower;
    // ensure we are in bounds
    if (step_lower < 0 || step_upper >= x_nominal.shape(0))
      throw std::runtime_error("Time t is out of bounds for policy evaluation");
  }
};

class ReplanManager
{
  // Triggers a replan and invalidates the internal policy if one exists
  // Note, it is an error to call this if the state is not READY, and
  // it is up to the caller to verify that with `ready_to_replan()`
  void request_replan(
      size_t horizon, size_t project_steps, double cur_time, double dt,
      const LinearPolicy& current_policy);

  // Checks if the manager is in the READY state
  bool ready_to_replan();

  // Returns the start time for which a new policy is available, if no policy is
  // avialable, returns std::nullopt
  std::optional<double> get_available_plan_start_time();

  // This moves from the internal policy, invalidating it
  std::unique_ptr<LinearPolicy> get_policy();

 private:
  enum State
  {
    // Planner can accept a request_replan
    READY,
    // Planner is computing a new policy, and will not accept a request_replan
    REQUESTED
  };

  std::unique_ptr<LinearPolicy> new_policy;
  std::mutex replan_mutex;
  std::condition_variable replan_cv;
};

// This is just a helper function for the most common replanning loop logic. The
// loop will request replans as quickly as possible and update the user's policy
// pointer as new plans become available. Regardless, the loop returns the
// command from the most recent available policy, which may either be the
// currently supplied policy or the newly computed policy
// Usage: while(true) { u = replan_iterate(...); send_to_motors(u);}
Tensor<float, 1> replan_iterate(
    size_t horizon, size_t project_steps, double cur_time, double dt,
    std::unique_ptr<LinearPolicy>& cur_policy, ReplanManager& replan_manager)

{
}

}  // namespace sds