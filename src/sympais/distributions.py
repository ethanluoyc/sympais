import jax
from jax import device_put
from jax import lax
from jax import random
from jax.dtypes import canonicalize_dtype
import jax.numpy as np
from jax.scipy import stats
from jax.scipy.special import ndtr
from jax.scipy.special import ndtri
from numpyro import distributions
from numpyro.distributions import constraints
from numpyro.distributions import Distribution
from numpyro.distributions.util import promote_shapes
from numpyro.distributions.util import sum_rightmost
from numpyro.distributions.util import validate_sample


class IndependentDistribution:

  def __init__(self, components):
    self.components = components

  def log_prob(self, x):
    logp = 0.
    for name, rv in self.components.items():
      logp += rv.log_prob(x[name])
    return jax.tree_util.tree_reduce(np.add, logp)

  def sample(self, rng_key, sample_shape=()):
    # rngs = split_rng_as(rng_key, self.components)
    samples = type(self.components)()
    for k, site in self.components.items():
      rng_key, rng_subkey = jax.random.split(rng_key)
      samples[k] = site.sample(rng_subkey, sample_shape)
    return samples


class Mixture(Distribution):
  support = constraints.real

  def __init__(self, mixture_distribution, components_distribution):
    self.mixture_distribution = mixture_distribution
    self.components_distribution = components_distribution
    super().__init__(batch_shape=self.mixture_distribution.batch_shape)

  def sample(self, rng_key, sample_shape):
    if isinstance(sample_shape, (tuple, list)) and len(sample_shape) > 1:
      raise NotImplementedError
    if len(sample_shape) == 0:
      n_samples = 1
    else:
      n_samples = sample_shape[0]
    x = self._sample_n(rng_key, n_samples)
    if len(sample_shape) == 0:
      x = x[0]
    return x

  def sample_with_intermediates(self, key, sample_shape=()):
    """
        Same as ``sample`` except that any intermediate computations are
        returned (useful for `TransformedDistribution`).

        :param jax.random.PRNGKey key: the rng_key key to be used for
          the distribution.
        :param tuple sample_shape: the sample shape for the distribution.
        :return: an array of shape `sample_shape + batch_shape + event_shape`
        :rtype: numpy.ndarray
        """
    return self.sample(key, sample_shape=sample_shape), []

  def _sample_n(self, rng, n):
    num_components = len(self.mixture_distribution.probs)
    cluster = self.mixture_distribution.sample(rng, (n,)).reshape(-1, 1)  # N, 1
    onehot_mask = cluster == np.arange(num_components)  # N, K
    onehot_mask = np.reshape(
        onehot_mask,
        onehot_mask.shape + (1,) * len(self.components_distribution.event_shape),
    )  # N, K, 1
    x = self.components_distribution.sample(rng, (n,))  # N, K, D #.shape
    return np.sum(x * onehot_mask, 1)

  def _pad_sample_dims(self, x, event_ndims=None):
    if event_ndims is None:
      event_ndims = len(self.components_distribution.event_shape)
    ndims = x.ndim
    shape = x.shape
    d = ndims - event_ndims
    x = np.reshape(x, (shape[:d] + (1,) + shape[d:]))
    return x

  def log_prob(self, x):  # N, D
    x = self._pad_sample_dims(x)
    log_prob_x = self.components_distribution.log_prob(x)  # [B, k]
    num_components = len(self.mixture_distribution.probs)
    log_mix_prob = np.log(self.mixture_distribution.probs).reshape(-1, num_components)
    return jax.scipy.special.logsumexp(log_prob_x + log_mix_prob, axis=-1)  # [S, B]


class TruncatedNormal(Distribution):
  """Truncated Normal implementation with both high and low"""

  arg_constraints = {
      "low": constraints.real,
      "loc": constraints.real,
      "scale": constraints.positive,
  }
  reparametrized_params = ["low", "loc", "scale"]

  # TODO: support `high` arg
  def __init__(self, low=0.0, high=1, loc=0.0, scale=1.0, validate_args=None):
    self.low, self.high, self.loc, self.scale = promote_shapes(low, high, loc, scale)
    self._normal = Normal(self.loc, self.scale)
    super().__init__(batch_shape=np.shape(self.loc), validate_args=validate_args)

  def sample(self, key, sample_shape=()):
    size = sample_shape + self.batch_shape
    # We use inverse transform method:
    # z ~ icdf(U), where U ~ Uniform(0, 1).
    u = random.uniform(key, shape=size)
    # Ref: https://en.wikipedia.org/wiki/Truncated_normal_distribution#Simulating
    # icdf[cdf_a + u * (1 - cdf_a)] = icdf[1 - (1 - cdf_a)(1 - u)]
    #                                 = - icdf[(1 - cdf_a)(1 - u)]
    #         return self.base_loc - ndtri(ndtr(self.base_loc) * (1 - u))
    alpha = (self.low - self.loc) / self.scale
    beta = (self.high - self.loc) / self.scale
    return ndtri(ndtr(alpha) + u * (ndtr(beta) - ndtr(alpha))) * self.scale + self.loc

  @validate_sample
  def log_prob(self, value):
    # log(cdf(high) - cdf(low)) = log(1 - cdf(low)) = log(cdf(-low))
    # log_ndtr(self.base_loc)
    alpha = (self.low - self.loc) / self.scale
    beta = (self.high - self.loc) / self.scale
    return self._normal.log_prob(value) - np.log(ndtr(beta) - ndtr(alpha))

  def cdf(self, value):
    alpha = (self.low - self.loc) / self.scale
    beta = (self.high - self.loc) / self.scale
    e = (value - self.loc) / self.scale
    normalizer = ndtr(beta) - ndtr(alpha)
    return (ndtr(e) - ndtr(alpha)) / normalizer


#     @property
#     def mean(self):
# #         low_prob_scaled = np.exp(self.base_dist.log_prob(0.))
# #         return self.loc + low_prob_scaled * self.scale

#     @property
#     def variance(self):
#         low_prob_scaled = np.exp(self.base_dist.log_prob(0.))
#         return (self.scale ** 2) *
#           (1 - self.base_dist.base_loc * low_prob_scaled - low_prob_scaled ** 2)


class Normal(distributions.Normal):

  def cdf(self, x):
    return stats.norm.cdf(x, loc=self.loc, scale=self.scale)


class Delta(Distribution):
  arg_constraints = {"value": constraints.real, "log_density": constraints.real}
  support = constraints.real
  is_discrete = True

  def __init__(self, value=0.0, log_density=0.0, event_ndim=0, validate_args=None):
    if event_ndim > np.ndim(value):
      raise ValueError("Expected event_dim <= v.dim(), actual {} vs {}".format(
          event_ndim, np.ndim(value)))
    batch_dim = np.ndim(value) - event_ndim
    batch_shape = np.shape(value)[:batch_dim]
    event_shape = np.shape(value)[batch_dim:]
    self.value = lax.convert_element_type(value, canonicalize_dtype(np.int32))
    # NB: following Pyro implementation,
    # log_density should be broadcasted to batch_shape
    self.log_density = promote_shapes(log_density, shape=batch_shape)[0]
    super().__init__(batch_shape, event_shape, validate_args=validate_args)

  def sample(self, key, sample_shape=()):
    del key
    shape = sample_shape + self.batch_shape + self.event_shape
    return np.broadcast_to(device_put(self.value), shape)

  @validate_sample
  def log_prob(self, value):
    log_prob = np.log(value == self.value)
    log_prob = sum_rightmost(log_prob, len(self.event_shape))
    return log_prob + self.log_density
