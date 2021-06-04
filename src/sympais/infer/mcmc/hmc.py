import collections

import jax
from jax import lax
from jax import random
from jax import tree_util
import jax.numpy as np
from numpyro import distributions

from sympais.infer import mcmc
from sympais.infer import utils

IntegratorState = collections.namedtuple(
    "IntegratorState", "state, state_grads, target_log_prob, momentum")


def leapfrog_step(state, target_log_prob_fn, kinetic_energy_fn, step_size, rng=None):
  """Single step of leapfrog.

    Notes
    =====

    The canonical distribution is related to the energy of the system
    by

    p(p, \theta) = 1/Zexp(-H(\theta, p)/T)

    For now, we assume that the kinetic energy takes
    the form
    K(p) = sum_i(p_i^2/(2m_i))
    """
  del rng
  p, q, q_grad = state.momentum, state.state, state.state_grads
  p_half = tree_util.tree_multimap(lambda p, qg: p + 0.5 * step_size * qg, p, q_grad)
  _, grad_p_half = utils.call_fn_value_and_grad(kinetic_energy_fn, p_half)
  q_full = tree_util.tree_multimap(lambda q, ph: q + step_size * ph, q, grad_p_half)
  logprob, q_full_grad = utils.call_fn_value_and_grad(target_log_prob_fn, q_full)
  p_full = tree_util.tree_multimap(lambda ph, qg: ph + 0.5 * step_size * qg, p_half,
                                   q_full_grad)
  return IntegratorState(q_full, q_full_grad, logprob, p_full)


def integrator_step(
    state,
    target_log_prob_fn,
    integrator_step_fn,
    kinetic_energy_fn,
    num_steps,
    step_size,
    rng,
):
  max_num_steps = np.max(num_steps)

  def body_fn(i, args):
    old_state, rng = args
    rng, rng_step = random.split(rng)
    new_state = integrator_step_fn(
        old_state,
        target_log_prob_fn,
        kinetic_energy_fn,
        step_size=step_size,
        rng=rng_step,
    )
    new_state = utils.choose((i < num_steps), new_state, old_state)
    return (new_state, rng)

  state, _ = lax.fori_loop(0, max_num_steps, body_fn, (state, rng))
  return state._replace(momentum=tree_util.tree_map(lambda x: -x, state.momentum))


_momentum_dist = distributions.Normal()


def sample_momentum(rng, state):
  """Sample momentum p for the system"""
  rngs = utils.split_rng_as(rng, state)
  return tree_util.tree_multimap(
      lambda s, r: _momentum_dist.sample(r, sample_shape=s.shape), state, rngs)


def gaussian_kinetic_energy_fn(*state, chain_ndims=0):
  # TODO: customize this when sampling momentum is also custimizable
  # ke = tree_util.tree_map(lambda s: -np.sum(_momentum_dist.log_prob(s)),
  #                         state)
  ke = tree_util.tree_map(
      lambda s: np.sum(np.square(s), axis=list(range(chain_ndims, s.ndim))) / 2, state)
  return tree_util.tree_reduce(lambda a, b: a + b, ke)


def hamiltonian_monte_carlo_step(
    rng,
    target_log_prob_fn,
    current_state,
    kinetic_energy_fn=gaussian_kinetic_energy_fn,
    integrator_step_fn=leapfrog_step,
    sample_momentum_fn=sample_momentum,
    num_steps=10,
    step_size=1.0,
):
  """ rng: random key
    state: initial state for parameters
    """
  rng, rng_momentum, rng_loguniform, rng_integrate = random.split(rng, 4)
  num_steps = np.asarray(num_steps)
  step_size = np.asarray(step_size)

  start_momentum = sample_momentum_fn(rng=rng_momentum, state=current_state)

  logprob, state_grads = utils.call_fn_value_and_grad(target_log_prob_fn, current_state)
  chain_ndims = logprob.ndim
  kinetic_energy_fn = jax.partial(kinetic_energy_fn, chain_ndims=chain_ndims)
  start_kinetic_energy = utils.call_fn(kinetic_energy_fn, start_momentum)
  assert start_kinetic_energy.shape == logprob.shape
  integrator_state = IntegratorState(current_state, state_grads, logprob,
                                     start_momentum)

  final_integrator_state = integrator_step(
      integrator_state,
      target_log_prob_fn,
      integrator_step_fn,
      kinetic_energy_fn,
      num_steps,
      step_size,
      rng=rng_integrate,
  )
  proposed_state = final_integrator_state.state
  proposed_kinetic_energy = utils.call_fn(kinetic_energy_fn,
                                          final_integrator_state.momentum)

  # HMC accepts with probability
  # alpha = min(1, exp(-U(q*)+U(q)-K(q*)+K(p)))
  # this is equivalent to
  #   u ~ U[0, 1]
  #   log(u) < exp(-U(q*)+U(q)-K(p*)+K(p))
  #   log(u) < exp(-U(q*)+U(q)-K(p*)+K(p))
  #   log(u) < (pi(q*)-pi(q) + (K(p) - K(p*)))
  #   <=> log(u) < (pi(q*)-pi(q) + (K(p) - K(p*)))
  # Check Metropolis acceptance criterion
  start_energy = (-utils.call_fn(target_log_prob_fn, current_state)
                  + start_kinetic_energy)
  new_energy = (-utils.call_fn(target_log_prob_fn, proposed_state)
                + proposed_kinetic_energy)
  log_acceptance_ratio = start_energy - new_energy
  return mcmc.metropolis_hasting_step(rng_loguniform, proposed_state, current_state,
                                      log_acceptance_ratio)
