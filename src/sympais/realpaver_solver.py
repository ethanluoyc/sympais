"""Solver using RealPaver for finding interval boxes from constraints."""
from typing import List, Optional

import numpy as np
from sympy.core import S
from sympy.printing.precedence import precedence
from sympy.printing.str import StrPrinter

from sympais import constraint
from sympais import types
from sympais.realpaver import interval_propagation
from sympais.realpaver import realpaver_api as api


class RealPaverCodePrinter(StrPrinter):
  """Printer that converts SymPy Expr to RealPaver expresssion.

  Note:
    The current implementation works nicely for +,-,*,/,^n,
    and (possibly) trigonometirc functions.
    In the future, we can extend this to work with a larger set of functions.
  """

  def _print_Pow(self, expr, rational=False):
    # Override pow since the StrPrinter uses x**2 instead of x^2.
    # This is adapted from the implementation from StrPrinter,
    # with the addition to disable using pow in producing results
    # and changing the expressed produced from x**n to x^n, the syntax expected
    # by RealPaver.
    PREC = precedence(expr)

    if expr.exp is S.Half and not rational:
      return "sqrt(%s)" % self._print(expr.base)

    if expr.is_commutative:
      if -expr.exp is S.Half and not rational:
        # Note: Don't test "expr.exp == -S.Half" here, because that will
        # match -0.5, which we don't want.
        return "%s/sqrt(%s)" % tuple(
            map(lambda arg: self._print(arg), (S.One, expr.base)))
      if expr.exp is -S.One:
        # Similarly to the S.Half case, don't test with "==" here.
        return "%s/%s" % (self._print(
            S.One), self.parenthesize(expr.base, PREC, strict=False))

    e = self.parenthesize(expr.exp, PREC, strict=False)
    if (self.printmethod == "_sympyrepr" and expr.exp.is_Rational and expr.exp.q != 1):
      # the parenthesized exp should be '(Rational(a, b))' so strip parens,
      # but just check to be sure.
      if e.startswith("(Rational"):
        return "%s^%s" % (self.parenthesize(expr.base, PREC, strict=False), e[1:-1])
    return "%s^%s" % (self.parenthesize(expr.base, PREC, strict=False), e)


def print_realpaver(expr: types.Constraint) -> str:
  return RealPaverCodePrinter().doprint(expr)


def build_realpaver_input(constraints: List[types.Constraint],
                          domains: Optional[types.DomainMap] = None,
                          timeout: Optional[float] = None) -> str:
  """Build input that can be used in RealPaver."""
  rp = api.RealPaverInput(timeout=timeout)
  variables = constraint.find_variables(constraints)
  if domains is None:
    domains = {v: (-np.inf, np.inf) for v in variables}
  for var in variables:
    rp.add_variable(var, domains[var.name][0], domains[var.name][1], var_type="real")

  for c in constraints:
    c = print_realpaver(c)
    rp.add_constraint(c)
  return rp.render()


class RealPaverSolver:
  """Solver using RealPaver for finding interval boxes from constraints."""

  def solve(self,
            constraints: types.Constraints,
            domains: types.DomainMap,
            precision: float = 0.01) -> interval_propagation.BoxList:
    input_file = build_realpaver_input(constraints, domains)
    boxes = interval_propagation.find_boxes(
        input_file, precision=precision, bin_path="realpaver")
    return boxes
