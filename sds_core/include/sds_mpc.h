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

// This class is responsible for safely managing a thread that computes new
// policies in the background. The user can request replans which are dispatched
// to the background thread, and query for new policies as they become
// available.
template <RolloutProvider<float> P, typename PolicyOptimizer>
class ReplanManager
{
  ReplanManager(const P& plant, const PolicyOptimizer& p)
      : plant(plant), optimizer(p)
  {
    std::thread replan_loop(
        [&]()
        {
          while (true)
          {
            std::unique_lock<std::mutex> lock(replan_mutex);
            replan_cv.wait(lock, [&]() { return state == REQUESTED; });
            lock.unlock();
            auto policy = replan();
            lock.lock();
            new_policy = std::move(policy);
            state = READY;
          }
        });
    replan_loop.detach();
  }

  // Triggers a replan and invalidates the internally-held policy if one exists
  // Note, it is an error to call this if the state is not READY, and
  // it is up to the caller to verify that with `ready_to_replan()`
  void request_replan(
      size_t horizon, size_t project_steps, double cur_time,
      const TensorView<float, 1>& x0, const LinearPolicy& current_policy)
  {
    replan_mutex.lock();
    if (state != READY)
      throw std::runtime_error("Called request_replan when not READY");
    replan_mutex.unlock();
    current_request.horizon = horizon;
    current_request.project_steps = project_steps;
    current_request.request_time = cur_time;
    current_request.x0 = x0;
    current_request.current_policy = current_policy;
    state = REQUESTED;
    replan_cv.notify_one();
  }

  std::unique_ptr<LinearPolicy> replan()
  {
    // Project ahead
    auto [x_proj, u_proj] = simulate_plant_with_policy(
        plant, current_request.current_policy, current_request.x0.view(),
        current_request.dt, current_request.request_time,
        current_request.horizon);

    // Optimize policy from the projected state
    return std::make_unique<LinearPolicy>(
        optimizer(x_proj.slice_1d<0>(current_request.project_steps)));
  }

  // Checks if the manager is in the READY state
  bool ready_to_replan()
  {
    std::lock_guard<std::mutex> lock(replan_mutex);
    return state == READY;
  }

  // Returns the start time for which a new policy is available, if no policy is
  // avialable, returns std::nullopt
  std::optional<double> get_available_plan_start_time()
  {
    std::lock_guard<std::mutex> lock(replan_mutex);
    if (new_policy)
      return std::make_optional(new_policy->start_time);
    else
      return std::nullopt;
  }

  // This moves from the internal policy, invalidating it
  std::unique_ptr<LinearPolicy> get_policy()
  {
    std::lock_guard<std::mutex> lock(replan_mutex);
    if (state != READY)
      throw std::runtime_error("Called get_policy when not READY");
    if (!new_policy)
      throw std::runtime_error(
          "Called get_policy when no new policy is available");
    return std::move(new_policy);
  }

 private:
  enum State
  {
    // Planner can accept a request_replan
    READY,
    // Planner is computing a new policy, and will not accept a request_replan
    REQUESTED
  };

  struct ReplanRequest
  {
    size_t horizon;
    size_t project_steps;
    double request_time;
    Tensor<float, 1> x0;
    LinearPolicy current_policy;
  };

  std::unique_ptr<LinearPolicy> new_policy;
  std::mutex replan_mutex;
  std::condition_variable replan_cv;
  State state = READY;

  // internal state for the replan request
  ReplanRequest current_request;
  PolicyOptimizer optimizer;
  P plant;
};

// This is just a helper function for the most common replanning loop logic. The
// loop will request replans as quickly as possible and update the user's policy
// pointer as new plans become available. Regardless, the loop returns the
// command from the most recent available policy, which may either be the
// currently supplied policy or the newly computed policy
// Usage: while(true) { u = replan_iterate(...); send_to_motors(u);}
template <typename P, PolicyOptimizer>
std::vector<float> replan_iterate(
    size_t horizon, size_t project_steps, double cur_time, double dt,
    const TensorView<float, 1>& x_cur,
    std::unique_ptr<LinearPolicy>& cur_policy,
    ReplanManager<P, PolicyOptimizer>& replan_manager)

{
  // Check if a new plan is available and swap over if it is and the start time
  // is in the past
  std::optional<double> maybe_available_plan_start_time =
      replan_manager.get_available_plan_start_time();

  if (maybe_available_plan_start_time &&
      *maybe_available_plan_start_time <= cur_time)
  {
    // swap over to the new policy
    cur_policy = std::move(replan_manager.get_policy());
  }

  // Request a new replan if we can
  if (replan_manager.ready_to_replan())
  {
    replan_manager.request_replan(
        horizon, project_steps, cur_time, dt, x_cur, *cur_policy);
  }
  // Evaluate the current policy at the current time
  return cur_policy->eval_policy(cur_time, x_cur.data());
}

}  // namespace sds