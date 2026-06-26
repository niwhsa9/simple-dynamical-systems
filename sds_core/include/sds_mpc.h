#pragma once
#include <concepts>

template <typename P, typename Scalar>
concept Policy = requires(P policy, Scalar t, Scalar* x) {
  { policy(t, x) };
};

struct LinearPolicy {

}

class ReplanManager {

  // Triggers a replan and invalidats the internal policy
  // Note, it is an error to call this if the state is not READY, and
  // it is up to the caller to verify that with `ready_to_replan()`
  void request_replan(
      size_t T, size_t project_steps, size_t k,
      BatchVortexModelManager& vmm_actual, const LinearPolicy& policy);

  void request_replan(
      size_t T, size_t project_steps, size_t k,
      BatchFlatPlateManager& fpm_actual, const LinearPolicy& policy);

  // Checks if the manager is in the READY state
  bool ready_to_replan();

  // Returns the start time for which a new policy is available, if no policy is
  // available, returns -1
  int get_available_plan_start_time();

  // This moves from the internal policy, invalidating it
  std::unique_ptr<LinearPolicy> get_policy();

 private:
  enum State
  {
    READY,
    REQUESTED
  };

  std::unique_ptr<LinearPolicy> new_policy;
  int new_policy_start_time;
  std::mutex replan_mutex;
  std::condition_variable replan_cv;

}