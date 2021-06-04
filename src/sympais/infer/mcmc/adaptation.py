import collections

import jax.numpy as np

# From
# https://colindcarroll.com/2019/04/21/step-size-adaptation-in-hamiltonian-monte-carlo/
DualAverageState = collections.namedtuple(
    "DualAverageState",
    [
        "mu",
        "target_accept",
        "gamma",
        "t",
        "kappa",
        "error_sum",
        "log_step",
        "log_averaged_step",
    ],
)


def dual_average_init(initial_step_size,
                      target_accept=0.65,
                      gamma=0.05,
                      t0=10.0,
                      kappa=0.75):
  return DualAverageState(
      mu=np.log(10 * initial_step_size),
      # proposals are biased upwards to stay away from 0.
      target_accept=target_accept,
      gamma=gamma,
      t=t0,
      kappa=kappa,
      error_sum=np.zeros_like(initial_step_size),
      log_step=np.log(initial_step_size),
      log_averaged_step=np.zeros_like(initial_step_size),
  )


def dual_average_step(state: DualAverageState, p_accept):
  # Running tally of absolute error. Can be positive or negative. Want to be 0.
  # This is the next proposed (log) step size. Note it is biased towards mu.
  error_sum = state.error_sum + state.target_accept - p_accept
  log_step = state.mu - error_sum / (np.sqrt(state.t) * state.gamma)
  # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
  eta = state.t**-state.kappa
  # Smoothed average step size
  log_averaged_step = eta * log_step + (1 - eta) * state.log_averaged_step
  # Return both the noisy step size, and the smoothed step size
  return DualAverageState(
      mu=state.mu,
      # proposals are biased upwards to stay away from 0.
      target_accept=state.target_accept,
      gamma=state.gamma,
      # This is a stateful update, so t keeps updating
      t=state.t + 1,
      kappa=state.kappa,
      error_sum=error_sum,
      log_step=log_step,
      log_averaged_step=log_averaged_step,
  )
