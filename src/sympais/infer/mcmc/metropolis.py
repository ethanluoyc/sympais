import collections

import jax
import jax.numpy as np
from numpyro import distributions

from sympais.infer import utils


def random_walk_proposal_fn(rng, state, scale=1.0):
  rngs = utils.split_rng_as(rng, state)
  proposed_state = jax.tree_multimap(
      lambda s, r: distributions.Normal(loc=s, scale=scale).sample(r), state, rngs)
  return proposed_state, 0.0


MetropolisHastingsState = collections.namedtuple("MetropolisHastingsState",
                                                 "state, log_prob")

MetropolisHastingsExtra = collections.namedtuple(
    "MetropolisHastingsExtra", "is_accepted, log_uniform, log_acceptance_ratio")


def metropolis_hasting_init(target_log_prob_fn, state):
  log_prob = utils.call_fn(target_log_prob_fn, state)
  return MetropolisHastingsState(state, log_prob)


def metropolis_hasting_step(rng, proposed_state, state, log_acceptance_ratio):
  logu = np.log(jax.random.uniform(rng, shape=log_acceptance_ratio.shape))

  is_accepted = logu < log_acceptance_ratio
  extra = MetropolisHastingsExtra(is_accepted, logu, log_acceptance_ratio)
  return utils.choose(is_accepted, proposed_state, state), extra


def random_walk_metropolis_hasting_step(rng, state, target_log_prob_fn, proposal_fn):
  state_log_prob = utils.call_fn(target_log_prob_fn, state)
  rng, rng_logu = jax.random.split(rng)
  proposed_state, log_proposed_bias = proposal_fn(rng, state)

  proposed_target_log_prob = utils.call_fn(target_log_prob_fn, proposed_state)

  assert state_log_prob.shape == proposed_target_log_prob.shape
  log_acceptance_ratio = (proposed_target_log_prob - state_log_prob - log_proposed_bias)

  return metropolis_hasting_step(rng_logu, proposed_state, state, log_acceptance_ratio)
