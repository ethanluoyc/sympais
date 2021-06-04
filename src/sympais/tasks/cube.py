"""Task for testing constraint satisfaction for a 2D polytope."""
import sympy

from sympais import distributions as dist

from . import base


class Cube2d(base.Task):
  """The cube2d task."""

  def __init__(self):
    b = 1.0
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    c1 = x + y <= b   # type: sympy.Expr
    c2 = x + y >= -b  # type: sympy.Expr
    c3 = y - x >= -b  # type: sympy.Expr
    c4 = y - x <= b  # type: sympy.Expr
    profile = {
        "x": dist.TruncatedNormal(low=-10., high=10., loc=-2, scale=1),
        "y": dist.TruncatedNormal(low=-10., high=10., loc=-2, scale=1)
    }
    domains = {"x": (-10., 10.), "y": (-10., 10.)}
    super().__init__(profile, [c1, c2, c3, c4], domains)
