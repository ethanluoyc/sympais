"""Importance sampling
This module implements variants of Importance Sampling (IS). It includes
  - basic importance sampling
  - mixture importance sampling
  - PI-MAIS
"""
import collections
from typing import Any, Tuple

import jax
from jax import random
from jax import tree_util
import jax.numpy as np
from jax.scipy.special import logsumexp

from sympais.infer import utils
from sympais.infer.mcmc import metropolis

ImportanceSamplingResult = collections.namedtuple("ImportanceSamplingResult", "x logw")


def _pad_last_dims(logp, x):
  """Broadcast logp along the right-most dimensions to match x."""
  event_dim = len(x.shape[2:])
  for _ in range(event_dim):
    logp = logp[..., None]
  return logp


def logmeanexp(a, axis=None):
  """Compute the logmeanexp of an array"""
  out = logsumexp(a, axis=axis)
  if axis is None:
    axis = np.arange(a.ndim)
  n = np.prod(np.take(a.shape, axis))
  return out - np.log(n)


def compute_ess(logw):
  """Compute Effective Sample Size (ESS) using log importance weights."""
  return 1 / np.exp(logsumexp(2 * (logw - logsumexp(logw))))


def logmean_logvar(logw):
  """Compute the log sample mean and variance of the importance sampling weights"""
  log_mean = logmeanexp(logw)
  log_mean_squared = logmeanexp(2 * logw)
  max2 = np.max(np.array([log_mean_squared, log_mean * 2]))
  logvar = (
      np.log(np.exp(log_mean_squared - max2) - np.exp(log_mean * 2 - max2)) + max2)
  return log_mean, logvar


class Proposal:
  """Proposal distribution for importance sampling (IS)"""

  def __init__(self, sample_fn, log_prob_fn):
    self.sample_fn = sample_fn
    self.log_prob_fn = log_prob_fn

  def sample(self, rng, sample_shape=()):
    return self.sample_fn(rng, sample_shape=sample_shape)

  def log_prob(self, *state):
    return self.log_prob_fn(*state)


def importance_sampling(rng, target_log_prob_fn, proposal,
                        n_sample: int) -> ImportanceSamplingResult:
  """Basic importance sampling."""
  x = proposal.sample(rng, sample_shape=(n_sample,))
  target_log_prob = utils.call_fn(target_log_prob_fn, x)
  proposal_log_prob = utils.call_fn(proposal.log_prob, x)
  logw = target_log_prob - proposal_log_prob

  return ImportanceSamplingResult(x, logw)


def mixed_proposal_importance_sampling(
    rng,
    target_log_prob_fn,
    proposal,
    num_samples: int,
) -> ImportanceSamplingResult:
  """mixtuer importance sampling."""
  x = proposal.sample(rng, (num_samples,))  # M, J
  target_log_prob = utils.call_fn(target_log_prob_fn, x)
  x_reshaped = jax.tree_util.tree_map(lambda x: np.expand_dims(x, 2), x)
  proposal_log_prob = utils.call_fn(proposal.log_prob, x_reshaped)
  logphi = np.log(np.mean(np.exp(proposal_log_prob), -1))
  # logphi = logmeanexp(proposal_log_prob, -1)
  logw = target_log_prob - logphi
  # When both q(x) == 0 and p(x) == 0, then logw = -inf and w = 0
  logw = np.where((target_log_prob == -np.inf) & (logphi == -np.inf), -np.inf, logw)
  return ImportanceSamplingResult(x, logw)


def adapt_proposal_step(rng, proposal_state, target_log_prob_fn, proposal_fn):
  next_proposal_state, _ = metropolis.random_walk_metropolis_hasting_step(
      rng, proposal_state, target_log_prob_fn, proposal_fn)
  return next_proposal_state


PIMAISState = collections.namedtuple(
    "PIMAISState", "Stot, Itot, Ipart, Ztot, Zpart, Zvar, t, proposal_state")


def pimais_init(proposal_state) -> PIMAISState:
  """Create initial state for PI-MAIS"""
  Itot = tree_util.tree_map(lambda x: np.zeros(x.shape[1:]), proposal_state)
  Ipart = tree_util.tree_map(lambda x: np.zeros(x.shape[1:]), proposal_state)
  return PIMAISState(0.0, Itot, Ipart, 0.0, 0.0, 0.0, 0.0, proposal_state)


def pimais_step(rng, target_log_prob_fn, kernel, build_proposal_fn, num_samples: int,
                state: PIMAISState) -> Tuple[PIMAISState, Any]:
  """Run one step of PI-MAIS."""
  if isinstance(state.proposal_state, list):
    num_proposals = state.proposal_state[0].shape[0]
  elif isinstance(state.proposal_state, dict):
    num_proposals = state.proposal_state[list(state.proposal_state.keys())[0]].shape[0]
  else:
    num_proposals = state.proposal_state.shape[0]
  # Adapt MCMC chain states by running a step wit the transition kernel.
  rng_kernel, rng_propose = jax.random.split(rng)
  proposal_state, extra = kernel(rng_kernel, state.proposal_state)
  # Build the mixture proposal distribution.
  proposal = build_proposal_fn(proposal_state)
  # IS with the mixture proposal distribution
  x, logw = mixed_proposal_importance_sampling(rng_propose, target_log_prob_fn,
                                               proposal, num_samples)

  # Update running statistics
  w = np.exp(logw)
  St = np.sum(w)
  Stot = state.Stot
  Itot = state.Itot
  t = state.t

  logdenom = jax.scipy.special.logsumexp(logw, [0, 1])
  logp = logw - logdenom  # normalized weights

  Ipart = tree_util.tree_map(
      lambda xx: np.sum(np.exp(_pad_last_dims(logp, xx)) * xx, [0, 1]), x)
  Itot = tree_util.tree_multimap(lambda it, ip: (Stot * it + St * ip) / (Stot + St),
                                 Itot, Ipart)
  Zpart = np.exp(logmeanexp(logw))
  m = t * num_proposals * num_samples
  n = num_proposals * num_samples
  # https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
  # var_part = np.mean(np.square(np.exp(logw) - np.exp(logmeanexp(logw))))
  _, logvarpart = logmean_logvar(logw)
  var_part = np.exp(logvarpart)
  new_var = m / (m + n) * state.Zvar + n / (m + n) * var_part + (
      m * n / ((m + n)**2) * np.square(state.Ztot - Zpart))

  Stot += St
  Ztot = 1 / (t + 1) / num_proposals / num_samples * Stot

  return (PIMAISState(Stot, Itot, Ipart, Ztot, Zpart, new_var, t + 1,
                      proposal_state), (x, logw, extra))


def PIMAIS(
    rng,
    target_log_prob_fn,
    proposal_state,
    kernel,
    build_proposal_fn,
    num_iters=1000,
    num_samples=10,
    trace_fn=lambda state, aux: (state[0], aux),
):
  """Run PI-MAIS.
  This function combines the `pimais_init` and `pimais_step`.
  """
  # 1. Initialization
  state = pimais_init(proposal_state)

  def step_fn(state_):
    state, rng = state_
    rng, rng_step = random.split(rng)
    next_state, aux = pimais_step(rng_step, target_log_prob_fn, kernel,
                                  build_proposal_fn, num_samples, state)
    return (next_state, rng), aux

  # 2. Iterate
  (state, _), aux = utils.trace((state, rng), step_fn, num_iters, trace_fn)
  return state, aux
