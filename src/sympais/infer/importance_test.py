"""Tests for inference algorithms."""
import functools
import unittest

import jax
from jax import random
import jax.numpy as jnp
from numpyro import distributions

import sympais.distributions as D
from sympais.infer import importance
from sympais.infer import mcmc
from sympais.infer.mcmc import metropolis


def _metropolis_hasting_step(rng,
                             proposal_state,
                             target_log_prob_fn,
                             metropolis_proposal_scale=5):
  proposal_fn = functools.partial(
      metropolis.random_walk_proposal_fn, scale=metropolis_proposal_scale)
  next_proposal_state, extra = metropolis.random_walk_metropolis_hasting_step(
      rng, proposal_state, target_log_prob_fn, proposal_fn)
  return next_proposal_state, extra


class ImportanceSamplingTestCase(unittest.TestCase):

  def _test_seed(self):
    return random.PRNGKey(42)

  def testIS(self):
    target_dist = distributions.Normal()
    proposal = distributions.Normal()
    n_samples = 10
    x, logw = importance.importance_sampling(self._test_seed(), target_dist.log_prob,
                                             proposal, n_samples)
    self.assertEqual(x.shape, (n_samples,))
    self.assertEqual(x.shape, logw.shape)

  def testMIS(self):
    target_log_prob_fn = distributions.Normal().log_prob
    num_proposals = 3
    num_samples = 10
    state = jnp.ones((num_proposals,))
    proposal = distributions.Normal(loc=state, scale=2.0)

    x, logw = importance.mixed_proposal_importance_sampling(self._test_seed(),
                                                            target_log_prob_fn,
                                                            proposal, num_samples)
    self.assertEqual(x.shape, (num_samples, num_proposals))
    self.assertEqual(x.shape, logw.shape)

  def testPIMAIS(self):
    target_dist = distributions.MultivariateNormal(
        loc=jnp.zeros(2), covariance_matrix=jnp.eye(2))
    num_proposals = 10
    proposal_state = jax.random.uniform(
        self._test_seed(), shape=(num_proposals, 2), minval=2, maxval=3)

    def get_proposal(proposal_state):
      return distributions.MultivariateNormal(proposal_state, jnp.eye(2))

    def kernel(key, state):
      return mcmc.random_walk_metropolis_hasting_step(key, state, target_dist.log_prob,
                                                      mcmc.random_walk_proposal_fn)

    importance.PIMAIS(
        self._test_seed(),
        target_dist.log_prob,
        proposal_state,
        kernel,
        get_proposal,
        num_iters=2,
        num_samples=10,
    )

  def test_PIMAIS_mixture(self):
    num_proposals = 2
    num_iterations = 10
    num_samples_per_iteration = 10
    importance_proposal_scale = 0.5
    metropolis_proposal_scale = 5.

    rng = random.PRNGKey(0)

    N_MIXTURES = 5
    INPUT_DIM = 2

    MU = jnp.array([[-10, -10], [0, 16], [13, 8], [-9, 7], [14, -14]])

    COV = jnp.array([[2, 0.6, 0.6, 1], [2, -0.4, -0.4, 2], [2, 0.8, 0.8, 2],
                     [3, 0, 0, 0.5], [2, -0.1, -0.1,
                                      2]]).reshape(-1, INPUT_DIM, INPUT_DIM)

    target_dist = D.Mixture(
        distributions.Categorical(probs=jnp.ones(N_MIXTURES) / N_MIXTURES),
        distributions.MultivariateNormal(loc=MU, covariance_matrix=COV))

    # initialize the proposal state
    proposal_state = jax.random.uniform(
        rng, shape=(num_proposals, INPUT_DIM), minval=-4, maxval=4)

    kernel = functools.partial(
        _metropolis_hasting_step,
        metropolis_proposal_scale=metropolis_proposal_scale,
        target_log_prob_fn=target_dist.log_prob)

    def get_proposal(proposal_state):
      """Build the lower layer proposal distributions"""
      return distributions.MultivariateNormal(
          proposal_state,
          jnp.eye(2) * jnp.square(importance_proposal_scale))

    importance.PIMAIS(
        rng,
        target_dist.log_prob,
        proposal_state,
        kernel,
        get_proposal,
        num_iters=num_iterations,
        num_samples=num_samples_per_iteration,
    )


if __name__ == "__main__":
  unittest.main()
