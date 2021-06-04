"""Implementation of SYMPAIS with RMH kernel and HMC kernel."""

import functools
from typing import Callable, Optional

from absl import logging
import jax
from jax import lax
from jax import random
import jax.numpy as np

from sympais import constraint
from sympais import distributions
from sympais import initializer
from sympais import logger as logger_lib
from sympais import profiles
from sympais import realpaver_solver
from sympais import tasks
from sympais import types
from sympais.infer import importance
from sympais.infer import utils
from sympais.infer.mcmc import hmc
from sympais.infer.mcmc import metropolis
from sympais.infer.mcmc.adaptation import dual_average_init
from sympais.infer.mcmc.adaptation import dual_average_step

State = types.NestedArray
Extra = types.NestedArray
Params = types.NestedArray


def _random_walk_proposal_fn(
    rng: types.PRNGKey,
    state: State,
    scale=1.0,
    domains=None,
):
  rngs = utils.split_rng_as(rng, state)
  proposed_state = jax.tree_multimap(
      lambda s, d, r: distributions.TruncatedNormal(
          low=d[0], high=d[1], loc=s, scale=scale).sample(r),
      state,
      domains,
      rngs,
  )
  return proposed_state, 0.0


class TransitionKernel:
  """Base class for defining a MCMC transition kernel."""

  def init(self) -> Params:
    """Initialize the parameters used by the kernel."""
    raise NotImplementedError()

  def step(self, params: Params, key: types.PRNGKey, state: State):
    """Run one step of MCMC."""
    raise NotImplementedError()


class RandomWalkMetropolisKernel(TransitionKernel):
  """Random-Walk Metropolis-Hastings transition kernel."""

  def __init__(self, target_log_prob_fn, scale, domains):
    self._target_log_prob_fn = target_log_prob_fn
    self._scale = scale
    self._domains = domains

  def init(self):
    return self._scale

  def step(self, params: Params, key: types.PRNGKey, state: State):
    rmh_scale = params
    proposal_fn = functools.partial(
        _random_walk_proposal_fn, scale=rmh_scale, domains=self._domains)
    next_proposal_state, extra = metropolis.random_walk_metropolis_hasting_step(
        key, state, self._target_log_prob_fn, proposal_fn)
    return next_proposal_state, extra


class HMCKernel(TransitionKernel):
  """Hamiltonian Monte Carlo transition kernel."""

  def __init__(self, target_log_prob_fn, step_size, num_steps):
    self._target_log_prob_fn = target_log_prob_fn
    self._step_size = step_size
    self._num_steps = num_steps

  def init(self):
    return self._step_size

  def step(self, params: Params, key: types.PRNGKey, state: State):
    return hmc.hamiltonian_monte_carlo_step(
        key,
        self._target_log_prob_fn,
        state,
        step_size=params,
        num_steps=self._num_steps)


def _update_scales(scales, acceptance_rate):
  """
    acc_rate of shape (a1, a2, ...)

    Notes:
    Adapted from PyMC3
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:
    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    """
  ranges = np.array([0.001, 0.05, 0.2, 0.5, 0.75, 0.95])
  multipliers = np.array([0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10])
  # a = jnp.expand_dims(acc_rate, -1)) is shape (a1, a2, ..., 1)
  # ranges <= a : (a1, a2, ..., N)
  return (scales * multipliers[np.sum(
      (ranges <= np.expand_dims(acceptance_rate, -1)), -1)])


def _compute_acceptance_rate(log_acceptance_ratio):
  p_accept = np.exp(log_acceptance_ratio)
  # Clip acceptance probability between 0. and 1.
  p_accept = np.where(p_accept < 1.0, p_accept, 1.0)
  return p_accept


class WindowedScaleAdaptor:
  """Step-size parameter adaptor for the RMH kernel."""

  def __init__(self, kernel, window_size: int):
    self._window_size = window_size
    self._kernel = kernel

  def __call__(self, rng, scales, chain_state, num_steps):
    state = chain_state
    num_windows = num_steps // self._window_size

    rngs = random.split(rng, num_windows)
    for i in range(num_windows):
      rng_step = rngs[i]
      state, (_, extra) = sample_chain(self._kernel, scales, rng_step, state, num_steps)
      utils.block_until_ready(state)
      p_accept = _compute_acceptance_rate(extra.log_acceptance_ratio)
      window_avg_acceptance_rate = np.mean(p_accept, axis=0)
      scales = _update_scales(scales, window_avg_acceptance_rate)
    return scales, state, ()


class ProposalBuilder:
  """Build the lower layer proposal distributions"""

  _proposal_builder: Callable[[State], types.Array]

  def __init__(
      self,
      profile: profiles.Profile,
      domains: types.DomainMap,
      proposal_scale_multiplier: float,
      proposal_std_num_samples: int,
      rng: types.PRNGKey,
  ):
    self._profile = profile
    self._domains = domains
    self._proposal_scale_multiplier = proposal_scale_multiplier
    self._proposal_std_num_samples = proposal_std_num_samples
    if isinstance(self._profile, profiles.IndependentProfile):

      def proposal_builder(proposal_state):
        dists = jax.tree_multimap(
            lambda s, scale, d: distributions.TruncatedNormal(
                low=d[0],
                high=d[1],
                loc=s,
                scale=self._proposal_scale_multiplier * scale,
            ),
            proposal_state,
            {k: v.scale for k, v in self._profile.components.items()},
            self._domains,
        )
        return distributions.IndependentDistribution(dists)
    else:
      # Estimate the empirical std of the profile
      samples = self._profile.sample(
          rng, sample_shape=(self._proposal_std_num_samples,))
      sample_std = jax.tree_map(np.std, samples)
      logging.info("Estimated sample std. is %f", sample_std)

      def proposal_builder(proposal_state):
        dists = jax.tree_multimap(
            lambda s, std: distributions.Normal(
                # FIXME (need to figure out how to choose scale propoerly)
                loc=s,
                scale=self._proposal_scale_multiplier * std,
            ),
            proposal_state,
            sample_std,
        )
        return distributions.IndependentDistribution(dists)

    self._proposal_builder = proposal_builder

  def __call__(self, proposal_state):
    """Return a callable that, when given the parallel chain states,
    produces a distribution as the importance sampling proposal distribution
    """
    return self._proposal_builder(proposal_state)


def refine_domains(constraints: types.Constraints,
                   domains: types.DomainMap) -> types.DomainMap:
  """Use ICP to perform domain refinements.

  This function calls the RealPaver to get the interval boxes.
  We then take use the max of the upper bounds and min of the lower bounds to
  find an interval box which is used to over-approximate the domain of the input
  distribution.
  This interval box will be used by the MCMC kernel to construct a truncated
  normal distribution as opposed to unbounded Gaussians.
  """
  boxes = realpaver_solver.RealPaverSolver().solve(
      constraints, domains=domains, precision=1e-1)
  new_domains = domains.copy()
  for var in domains:
    updated_lower = min([b.intervals[var].lower for b in boxes if var in b.intervals])
    updated_upper = max([b.intervals[var].upper for b in boxes if var in b.intervals])
    old_domain = new_domains[var]
    new_domains[var] = (
        max(updated_lower, old_domain[0]),
        min(updated_upper, old_domain[1]),
    )
  return new_domains


@functools.partial(jax.jit, static_argnums=(0, 4))
def sample_chain(kernel, params, rng, state, num_steps):
  """Run parallel chain MCMC sampling."""

  def wrapped_fn(state, _unused):
    del _unused
    rng, state = state
    rng, rng_step = jax.random.split(rng)
    next_state, kernel_extra = kernel.step(params, rng_step, state)
    return (rng, next_state), (next_state, kernel_extra)

  (_, final_state), extra = lax.scan(
      wrapped_fn, (rng, state), xs=None, length=num_steps)
  return final_state, extra


def run_sympais(task: tasks.Task,
                key,
                num_samples=int(1e6),
                num_proposals=100,
                num_samples_per_iter=5,
                proposal_scale_multiplier=0.5,
                rmh_scale=1.0,
                init="z3",
                tune=True,
                num_warmup_steps=500,
                window_size=100,
                resample=True,
                proposal_std_num_samples=100,
                logger=None):
  """Run SYMPAIS with the RMH kernel.

  This functions is kept here for maintaining interface compatibility
  with the other benchmarking algorithms. You should refer to `sympais.run`
  for a SYMPAIS implementation that provides richer output.
  """
  profile = task.profile
  pcs = task.constraints
  constraint_fn = constraint.build_constraint_fn(pcs, task.domains)
  target_log_prob_fn = constraint.build_target_log_prob_fn(task.profile, task.domains,
                                                           constraint_fn)
  domains = task.domains
  domains = refine_domains(pcs, domains)

  key, subkey = jax.random.split(key)
  proposal_builder = ProposalBuilder(profile, domains, proposal_scale_multiplier,
                                     proposal_std_num_samples, subkey)
  initializer_ = initializer.Initializer(profile, pcs, domains, init, resample)
  kernel = RandomWalkMetropolisKernel(target_log_prob_fn,
                                      np.ones(num_proposals) * rmh_scale, domains)
  kernel.step = jax.jit(kernel.step)
  params = kernel.init()
  key, subkey = jax.random.split(key)
  chain_state = initializer_(num_proposals, subkey)

  key, subkey = jax.random.split(key)
  if num_warmup_steps < 1:
    logging.info("Not running warmup")
  else:
    if tune:
      params, chain_state, _ = WindowedScaleAdaptor(kernel,
                                                    window_size)(subkey, params,
                                                                 chain_state,
                                                                 num_warmup_steps)
    else:
      logging.info("Not tuning the kernel")
      chain_state, (_, extra) = sample_chain(kernel, params, subkey, chain_state,
                                             num_warmup_steps)
  logging.info("Finished warm-up")
  # Start running PIMAIS
  state = importance.pimais_init(chain_state)

  @jax.jit
  def pimais_step_fn(params, rng, state):
    kernel_fn = lambda key, state: kernel.step(params, key, state)
    return importance.pimais_step(rng, target_log_prob_fn, kernel_fn, proposal_builder,
                                  num_samples_per_iter, state)

  num_samples_warmup = num_proposals * num_warmup_steps
  num_iterations = (num_samples - num_samples_warmup) // (
      (num_proposals + 1) * num_samples_per_iter)
  rngs = jax.random.split(key, num_iterations)
  num_samples_used = num_samples_warmup
  for idx in range(0, num_iterations):
    # tic = time.time()
    state, extra = pimais_step_fn(params, rngs[idx], state)
    num_samples_used += num_samples_per_iter * (num_proposals + 1)
    # Make sure async dispatch is accounted for in measuring running time.
    utils.block_until_ready((state, extra))
    # toc = time.time()
    # print("Time elapsed", toc - tic, "Mean", state.Ztot)
    if logger is not None:
      logs = {"sample_count": num_samples_used, "mean": state.Ztot}
      logger.write(logs)
  return {"mean": state.Ztot}


def _hmc_step(
    rng,
    proposal_state,
    target_log_prob_fn,
    num_steps=10,
    step_size=0.1,
):
  next_state, extra = hmc.hamiltonian_monte_carlo_step(
      rng, target_log_prob_fn, proposal_state, step_size=step_size, num_steps=num_steps)
  return next_state, extra


class DualAveragingAdaptor:
  """Dual averaging adaptor for HMC step size."""

  def __init__(self, target_log_prob_fn, num_steps):
    self._target_log_prob_fn = target_log_prob_fn
    self._num_steps = num_steps

  def run(self, rng, kernel_params, proposal_state, num_steps):
    step_size = kernel_params

    # Tune the step size using dual averaging
    # Similar to PyMC tuning stratigies
    def adaptive_kernel(state_):
      proposal_state, rng, da_state = state_
      next_state, extra = _hmc_step(
          rng,
          proposal_state,
          target_log_prob_fn=self._target_log_prob_fn,
          step_size=step_size,
          num_steps=self._num_steps,
      )
      p_accept = np.exp(extra.log_acceptance_ratio)
      p_accept = np.where(p_accept < 1.0, p_accept, 1.0)
      da_state = dual_average_step(da_state, p_accept)
      return (next_state, rng, da_state), (p_accept, extra, da_state)

    state = (proposal_state, rng, dual_average_init(step_size))
    state, (p_accept, _, da_state) = utils.trace(state, adaptive_kernel, num_steps)
    logging.info("Average acceptance probability after tuning is %s", p_accept[-1])
    final_step_size = np.exp(state[-1].log_averaged_step)
    logging.info("Average step size is %s", final_step_size)
    return final_step_size, state[0], da_state


def run_sympais_hmc(task: tasks.Task,
                    key,
                    num_samples: int = int(1e6),
                    num_proposals: int = 100,
                    num_samples_per_iter: int = 5,
                    proposal_scale_multiplier: float = 0.5,
                    step_size: float = 0.1,
                    num_steps: int = 20,
                    num_warmup_steps: int = 500,
                    init: str = "realpaver",
                    resample: bool = True,
                    tune: bool = False,
                    proposal_std_num_samples: int = 100,
                    logger: Optional[logger_lib.Logger] = None):
  """Run SYMPAIS with the Hamiltonian Monte Carlo kernel."""
  profile = task.profile
  pcs = task.constraints
  constraint_fn = constraint.build_constraint_fn(pcs, task.domains)
  target_log_prob_fn = constraint.build_target_log_prob_fn(task.profile, task.domains,
                                                           constraint_fn)
  domains = task.domains
  domains = refine_domains(pcs, domains)

  key, subkey = jax.random.split(key)
  proposal_builder = ProposalBuilder(profile, domains, proposal_scale_multiplier,
                                     proposal_std_num_samples, subkey)
  initializer_ = initializer.Initializer(profile, pcs, domains, init, resample)
  # Scale the initial step size by 1/n**(1/4) as in
  # https://github.com/pymc-devs/pymc3/blob/ea1b03811f7d58a38c09a56a61144294f3732d71/pymc3/step_methods/hmc/base_hmc.py#L47
  # where `n` is the dimensionality of the input profile
  if isinstance(profile, profiles.IndependentProfile):
    num_dimensions = len(profile.components)
  else:
    num_dimensions = np.prod(np.asarray(profile.event_shape))
  step_size = np.full(num_proposals, step_size) / (num_dimensions ** 0.25)
  kernel = HMCKernel(target_log_prob_fn, np.ones(num_proposals) * step_size, num_steps)
  kernel.step = jax.jit(kernel.step)
  params = kernel.init()
  key, subkey = jax.random.split(key)
  chain_state = initializer_(num_proposals, subkey)
  sample_chain_fn = sample_chain

  key, subkey = jax.random.split(key)
  if num_warmup_steps < 1:
    logging.info("Not running warmup")
  else:
    # we have not been successful in enabling tuning in the HMC for a few reasons
    # we do not use a fixed path length but fix the number of integration steps;
    # we are running HMC on constrained spaces instead of R^d. Tuning becomes a bit
    # problematic in our setting.
    if tune:
      params, chain_state, _ = (
          DualAveragingAdaptor(target_log_prob_fn,
                               num_steps).run(subkey, params, chain_state,
                                              num_warmup_steps))
    else:
      logging.info("Not tuning the kernel")
      # TODO: tuning for HMC kernel is disabled.
      chain_state, (_, extra) = sample_chain_fn(kernel, params, subkey, chain_state,
                                                num_warmup_steps)
    logging.info("Finished warm-up")
  # Start running PIMAIS
  state = importance.pimais_init(chain_state)

  @jax.jit
  def pimais_step_fn(params, rng, state):
    kernel_fn = lambda key, state: kernel.step(params, key, state)
    return importance.pimais_step(rng, target_log_prob_fn, kernel_fn, proposal_builder,
                                  num_samples_per_iter, state)

  num_samples_warmup = num_proposals * num_warmup_steps
  num_iterations = (num_samples - num_samples_warmup) // (
      (num_proposals + 1) * num_samples_per_iter)
  rngs = jax.random.split(key, num_iterations)
  num_samples_used = num_samples_warmup
  for idx in range(0, num_iterations):
    # tic = time.time()
    state, extra = pimais_step_fn(params, rngs[idx], state)
    num_samples_used += num_samples_per_iter * (num_proposals + 1)
    # Make sure async dispatch is accounted for in measuring running time.
    utils.block_until_ready((state, extra))
    if logger is not None:
      logs = {"sample_count": num_samples_used, "mean": state.Ztot}
      logger.write(logs)
    # toc = time.time()
    # print("Time elapsed", toc - tic, "Mean", state.Ztot)
  return {"mean": state.Ztot}
