"""Test for SYMPAIS"""

from absl.testing import absltest
import sympy

from sympais import algorithm
from sympais import distributions
from sympais.tasks import Task


class SYMPAISTest(absltest.TestCase):

  def test_unsat_returns_zero_prob(self):
    profile = {"x": distributions.Normal()}
    x = sympy.Symbol("x")
    cs = (x * x >= 1, x * x <= -1)
    task = Task(profile, cs, domains={"x": (-10., 10.)})
    algorithm.sympais(task, 0)


if __name__ == "__main__":
  absltest.main()
