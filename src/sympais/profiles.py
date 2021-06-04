"""Program input profiles."""
import warnings

import jax
import jax.numpy as jnp


class Profile:
  """An input profile models the input distribution to a program.

  This is a synonym to an input distribution, so the interface
  follows that of a `numpyro.Distribution`.

  """

  def log_prob(self, event):
    """Returns the log probability of the event.
      Args:
        event: a dictionary from variable names to the sample values.

      Returns:
        log_prob: a jax ndarray of the log density evaluated at `event`
    """
    raise NotImplementedError()

  def sample(self, rng_key, sample_shape=()):
    """Draw samples from a input profile.
      Args:
        rng_key: `jax.random.PRNGKey` for the random number generator seed.
        sample_shape: tuple of the sample shape.
      Returns:
        samples: a dictionary of sampled inputs to the probabilistic program.
    """
    raise NotImplementedError()


class IndependentProfile(Profile):
  """An input profile where the variables are independent.

  Given a dictionary `components` for the independent component distribution,
  this class implements functions for sampling from the independent distributions
  and evaluating the pdf.
  """

  def __init__(self, components):
    for name, rv in components.items():
      if rv.is_discrete:
        warnings.warn(f"discrete distribution for {name} not supported", Warning)
    self._components = components

  @property
  def components(self):
    return self._components

  def __repr__(self):
    return "IndependentProfile({})".format(self._components)

  def log_prob(self, event):
    logp = 0.
    for name, rv in self.components.items():
      logp += rv.log_prob(event[name])
    return jax.tree_util.tree_reduce(jnp.add, logp)

  def sample(self, rng_key, sample_shape=()):
    # rngs = split_rng_as(rng_key, self.components)
    samples = type(self.components)()
    for k, site in self.components.items():
      rng_key, rng_subkey = jax.random.split(rng_key)
      samples[k] = site.sample(rng_subkey, sample_shape)
    return samples
