import jax
from numpyro.distributions import StudentT
from sympy import Symbol

from sympais import distributions as dist
from sympais import profiles

from . import base


class CorrelatedProfile(profiles.Profile):

  def __init__(self, scale):
    super().__init__()
    self.scale = scale

  def log_prob(self, s):
    x, y, z = s["x"], s["y"], s["z"]
    xdist = StudentT(df=2)
    ydist = dist.Normal(s["x"], self.scale)
    zdist = dist.Normal(s["x"], self.scale)
    return xdist.log_prob(x) + ydist.log_prob(y) + zdist.log_prob(z)

  @property
  def event_shape(self):
    return (3,)

  def sample(self, key, sample_shape=()):
    rngx, rngy, rngz = jax.random.split(key, 3)
    xdist = StudentT(df=2)
    x = xdist.sample(rngx, sample_shape)
    # Careful! Since x has sample_shape, the distribution for y, z broadcasts,
    # hence sample_shape = ()
    y = dist.Normal(x, self.scale).sample(rngy, ())
    z = dist.Normal(x, self.scale).sample(rngz, ())
    return {"x": x, "y": y, "z": z}


class Torus(base.Task):

  def __init__(self, scale=0.5, profile_type="independent"):
    R = 3.0
    r = 1.0
    self._profile_type = profile_type
    self._scale = scale
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    profile = self._get_profile()
    # constraint = (
    #     primitives.square(primitives.sqrt(x * x + y * y) - R) + z * z <= r * r
    # )
    # values = {"x": 2., "y": 2., "z": 0.}
    # (x ** 2 + y ** 2 + z ** 2 + R ** 2 - r ** 2) ** 2
    #   <= (4 * R ** 2) * (x ** 2 + y ** 2)
    tmp = (x * x + y * y + z * z + R * R - r * r)
    constraint = (tmp * tmp <= (4 * R * R) * (x * x + y * y))
    domains = {"x": (-5, 5), "y": (-5, 5), "z": (-5, 5)}
    super().__init__(profile, (constraint,), domains)

  def _get_profile(self):
    scale = self._scale
    if self._profile_type == "independent":
      profile = {
          "x": dist.Normal(loc=0, scale=scale),
          "y": dist.Normal(loc=0, scale=scale),
          "z": dist.Normal(loc=0, scale=scale),
      }
    elif self._profile_type == "correlated":
      profile = CorrelatedProfile(scale)
    else:
      raise ValueError("Unknown profile_type: {}".format(self._profile_type))
    return profile
