"""Test solvers can find initial solutions."""
from absl.testing import absltest
from absl.testing import parameterized
from sympy import symbols

from sympais import realpaver_solver
from sympais import z3_solver


def _get_sphere_constraint():
  x, y = symbols("x y")
  sphere_expr = (x - 1) * (x - 1) + (y - 1)**2 <= 1
  return (sphere_expr,), {"x": (-10, 10.), "y": (-10., 10.)}


def _get_torus_constraint():
  x, y, z = symbols("x y z")
  R = 3.0
  r = 1.0

  # constraint = (
  #     primitives.square(primitives.sqrt(x * x + y * y) - R) + z * z <= r * r
  # )
  # values = {'x': 2., 'y': 2., 'z': 0.}
  # (x ** 2 + y ** 2 + z ** 2 + R ** 2 - r ** 2) ** 2 <=
  # (4 * R ** 2) * (x ** 2 + y ** 2)
  tmp = (x * x + y * y + z * z + R * R - r * r)
  constraint = (tmp * tmp <= (4.0 * R * R) * (x * x + y * y))
  domains = {"x": (-5.0, 5.0), "y": (-5.0, 5.0), "z": (-5.0, 5.0)}
  return (constraint,), domains


class Z3Test(parameterized.TestCase):

  @parameterized.parameters(
      _get_sphere_constraint(),
      _get_torus_constraint(),
  )
  def test_solve(self, constraints, domains):
    solution = z3_solver.Z3Solver().solve(constraints, domains)
    assert solution is not None


class RealPaverTest(parameterized.TestCase):

  @parameterized.parameters(
      _get_sphere_constraint(),
      _get_torus_constraint(),
  )
  def test_solve(self, constraints, domains):
    solution = realpaver_solver.RealPaverSolver().solve(constraints, domains)
    assert solution is not None


if __name__ == "__main__":
  absltest.main()
