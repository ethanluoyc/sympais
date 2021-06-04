"""Solver using Z3 for finding feasible solution from constraints."""
import sympy
import z3

from sympais import constraint
from sympais import types

# TODO: move to a configuration modules to separate
z3.set_option("smt.string_solver", "z3str3")


class UnsatConstraint(Exception):
  """Exception when the constraint is unsat."""
  pass


def z3_value_to_python(value):
  if z3.is_true(value):
    return True
  elif z3.is_false(value):
    return False
  elif z3.is_int_value(value):
    return value.as_long()
  elif z3.is_rational_value(value):
    return float(value.numerator_as_long()) / float(value.denominator_as_long())
  elif z3.is_string_value(value):
    return z3_string_decoder(value)
  elif z3.is_algebraic_value(value):
    raise NotImplementedError()
  else:
    raise NotImplementedError()


def z3_string_decoder(z3ModelString: z3.StringVal):
  length = z3.Int("length")
  tmp_string = z3.String("ts")
  solver = z3.Solver()
  solver.add(tmp_string == z3ModelString)
  solver.add(z3.Length(tmp_string) == length)
  assert solver.check() == z3.sat

  model = solver.model()
  assert model[length].is_int()
  num_chars = model[length].as_long()

  solver.push()
  char_bvs = []
  for i in range(num_chars):
    char_bvs.append(z3.BitVec("ch_%d" % i, 8))
    solver.add(z3.Unit(char_bvs[i]) == z3.SubString(tmp_string, i, 1))

  assert solver.check() == z3.sat
  model = solver.model()
  python_string = "".join([chr(model[ch].as_long()) for ch in char_bvs])
  return python_string


def sympy_to_z3(sympy_var_list, exprs):
  """Convert sympy expressions to a z3 expressions.

  Returns:
    a tuple (z3_vars, z3_expression),
    z3_var is a dict mapping from sympy `Symbols` to Z3 symbols and
    z3_expression is a list of Z3 expressions.
  """
  z3_var_map = {}

  for var in sympy_var_list:
    name = var.name
    z3_var = z3.Real(name)
    z3_var_map[name] = z3_var

  result_exps = [_sympy_to_z3_rec(z3_var_map, e) for e in exprs]
  return z3_var_map, result_exps


def _sympy_to_z3_rec(variable_map, expr):
  """Recursively convert sympy expression to z3 expressions."""
  # Adapted from
  # https://stackoverflow.com/questions/22488553/how-to-use-z3py-and-sympy-together

  rv = None

  # TODO(yl): For some reason,
  # GreaterThan and LessThan are not subclasses of Expr?
  # if not isinstance(e, Expr):
  #   raise RuntimeError("Expected sympy Expr: " + repr(e))

  if isinstance(expr, sympy.Symbol):
    rv = variable_map.get(expr.name)
    if rv is None:
      raise RuntimeError("No var was corresponds to symbol '" + str(expr) + "'")

  elif isinstance(expr, sympy.Number):
    rv = float(expr)
  elif isinstance(expr, sympy.Mul):
    rv = _sympy_to_z3_rec(variable_map, expr.args[0])

    for child in expr.args[1:]:
      rv *= _sympy_to_z3_rec(variable_map, child)
  elif isinstance(expr, sympy.Add):
    rv = _sympy_to_z3_rec(variable_map, expr.args[0])

    for child in expr.args[1:]:
      rv += _sympy_to_z3_rec(variable_map, child)
  elif isinstance(expr, sympy.Pow):
    term = _sympy_to_z3_rec(variable_map, expr.args[0])
    exponent = _sympy_to_z3_rec(variable_map, expr.args[1])

    if exponent == 0.5:
      # sqrt
      rv = z3.Sqrt(term)
    else:
      rv = term**exponent
  elif isinstance(expr, sympy.LessThan):
    lhs = _sympy_to_z3_rec(variable_map, expr.args[0])
    rhs = _sympy_to_z3_rec(variable_map, expr.args[1])
    rv = lhs <= rhs
  elif isinstance(expr, sympy.GreaterThan):
    lhs = _sympy_to_z3_rec(variable_map, expr.args[0])
    rhs = _sympy_to_z3_rec(variable_map, expr.args[1])
    rv = lhs >= rhs
  # TODO(yl): Support more expressions
  if rv is None:
    raise RuntimeError("Type '" + str(type(expr))
                       + "' is not yet implemented for convertion to a z3 expresion. "
                       + "Subexpression was '" + str(expr) + "'.")

  return rv


class Z3Solver:
  """Solver using Z3 for finding feasible solution from constraints."""

  def solve(self, constraints: types.Constraints, domains: types.DomainMap):
    variables = constraint.find_variables(constraints)
    # Convert domains to constraints
    var_map = {}
    for v in variables:
      var_map[v.name] = v

    domain_constraints = []
    for var_name, domain in domains.items():
      domain_constraints.extend([
          var_map[var_name] >= domain[0],
          var_map[var_name] <= domain[1],
      ])

    z3_var, cs = sympy_to_z3(variables, list(constraints) + list(domain_constraints))

    solver = z3.Solver()
    for c in cs:
      solver.add(c)  # add a constraint with converted expression

    result = solver.check()
    if result == z3.sat:
      m = solver.model()
      output = {}
      for var in variables:
        output[var.name] = z3_value_to_python(m[z3_var[var.name]])
      return output
    elif result == z3.unsat:
      raise UnsatConstraint()
    else:
      return None
