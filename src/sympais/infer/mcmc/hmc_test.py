import unittest

from jax import random
import jax.numpy as np

from sympais.infer.mcmc import hmc


class HMCTest(unittest.TestCase):

  def _test_seed(self):
    return random.PRNGKey(42)

  def test_sample_momentum(self):
    theta = np.zeros(3)
    p = hmc.sample_momentum(self._test_seed(), theta)
    self.assertEqual(p.shape, theta.shape)

  def test_sample_momentum_struct(self):
    theta = np.zeros(3)
    p = hmc.sample_momentum(self._test_seed(), theta)
    self.assertEqual(p.shape, theta.shape)


if __name__ == "__main__":
  unittest.main()
