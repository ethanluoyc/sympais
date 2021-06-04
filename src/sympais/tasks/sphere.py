"""Benchmark task for testing constraint satisfaction with sphere."""
import sympy

from sympais import distributions as dist

from . import base


class Sphere(base.Task):
  """Task for constraint satisfaction with spheres."""

  def __init__(self, nd: int):
    """Construct a `Sphere` task.
    Args:
      nd: number of dimensions for the sphere.
    """
    xs = [sympy.Symbol("x{}".format(i)) for i in range(nd)]
    # evalute=False Disables usage of x^n expressions in producing
    # RealPaver expressions.
    # This makes ICP less efficient, but is consistent with the usage
    # in the paper for producing interval boxes from the sphere benchmark.
    s = sum([sympy.Mul((x - 1), (x - 1), evaluate=False) for x in xs])
    c = s <= 1.0
    constraints = (c,)
    profile = {
        "x{}".format(i): dist.TruncatedNormal(low=-10., high=10., loc=0, scale=1)
        for i in range(nd)
    }
    domains = {"x{}".format(i): (-10., 10.) for i in range(nd)}
    super().__init__(profile, constraints, domains)
