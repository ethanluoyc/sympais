"""Tests for MCMC kernels"""
import unittest

from jax import random
import jax.numpy as jnp
from numpyro import distributions

from sympais.infer.mcmc import hmc
from sympais.infer.mcmc import metropolis


class MCMCTestCase(unittest.TestCase):

  def _test_seed(self):
    return random.PRNGKey(42)

  def test_metropolis_hasting(self):

    def target_log_prob_fn(x, y):
      return distributions.Normal().log_prob(x) + distributions.Normal().log_prob(y)

    state = [jnp.ones(()), jnp.ones(())]

    next_state, _ = metropolis.random_walk_metropolis_hasting_step(
        self._test_seed(), state, target_log_prob_fn,
        metropolis.random_walk_proposal_fn)
    self.assertEqual(next_state[0].shape, state[0].shape)
    self.assertEqual(next_state[1].shape, state[1].shape)

  def test_hmc_step(self):

    def target_log_prob_fn(x, y):
      return distributions.Normal().log_prob(x) + distributions.Normal().log_prob(y)

    state = [jnp.ones(()), jnp.ones(())]

    next_state, _ = hmc.hamiltonian_monte_carlo_step(self._test_seed(),
                                                     target_log_prob_fn, state)
    self.assertEqual(next_state[0].shape, state[0].shape)
    self.assertEqual(next_state[1].shape, state[1].shape)


if __name__ == "__main__":
  unittest.main()
