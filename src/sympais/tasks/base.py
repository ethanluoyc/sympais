"""Definition for a probabilistic program analysis task."""
from typing import Dict, Optional, Union

from numpyro import distributions

from sympais import profiles
from sympais import types


class Task:
  """Base class for a probabilistic program analysis task.

  Benchmark tasks should subclass `Task` and call the `super().__init__()`
  method with the task-specific profile, constraints and domains.
  `Task` allows the subclasses to provide a dictionary of `numpyro.Distribution`
  as the input profile. The dictionary will be converted to an `Independent` profile.

  Attributes:
    profile: the input profile for the task.
    constraints: the constraints for the task query.
    domains: the domains for the input profile.
  """

  def __init__(self,
               profile: Union[profiles.Profile, Dict[str, distributions.Distribution]],
               constraints: types.Constraints,
               domains: Optional[types.DomainMap] = None):
    if isinstance(profile, dict):
      self.profile = profiles.IndependentProfile(profile)
    else:
      self.profile = profile
    self.constraints = constraints
    self.domains = domains
