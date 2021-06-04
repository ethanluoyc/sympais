"""Functions for manipulating constraints."""
from typing import List, Set, Tuple, Union

import jax.numpy as jnp
import sympy
from sympy import core

from sympais import jax_printer
from sympais import profiles
from sympais import types


def find_variables(
    expr: Union[types.Constraint, List[types.Constraint], Tuple[types.Constraint]]
) -> Set[core.Symbol]:
  """Find variables in constraints."""
  if not isinstance(expr, (list, tuple)):
    expr = []
  variables = set()
  for e in expr:
    variables |= e.free_symbols
  return variables


def _build_constraint_fn(expr: types.Constraint) -> types.ConstraintFn:
  """Create a Python function for evaluating sympy expression with JAX."""
  # Convert to a list since the args need to maintain order
  free_symbols = list(expr.free_symbols)
  arg_names = [s.name for s in free_symbols]
  # This creates a function
  # `lambified_fun` that accepts a list of positional args and evaluate
  # it with jax.numpy
  # TODO(yl) handle jnp backend more robustly.
  lambdified_fun = jax_printer.jaxify(free_symbols, expr)

  # Wrapping the positional arg signature with a signature that supports
  # a single arg that is a dictionary of concrete values.
  def fun(subs):
    in_tree = [subs[name] for name in arg_names]
    return lambdified_fun(*in_tree)

  return fun


def domain_to_constraints(domains: types.DomainMap) -> List[types.ConstraintFn]:
  """Convert domains to constraints."""
  domain_constraints = []
  for var_name, domain in domains.items():
    domain_constraints.extend([
        core.Symbol(var_name) >= domain[0],
        core.Symbol(var_name) <= domain[1],
    ])
  return domain_constraints


def build_constraint_fn(constraints: Union[List[types.Constraint],
                                           Tuple[types.Constraint]],
                        domains: types.DomainMap) -> types.ConstraintFn:
  """Build a constraint function from a list of constraints and domains."""
  variables = find_variables(constraints)
  # Convert domains to constraints
  var_map = {}
  for v in variables:
    var_map[v.name] = v

  domain_constraints = domain_to_constraints(domains)
  cs = list(constraints) + list(domain_constraints)
  combined_constraint = sympy.And(*cs)
  return _build_constraint_fn(combined_constraint)


def build_target_log_prob_fn(
    profile: profiles.Profile,
    domains: types.DomainMap,
    constraint_fn: types.ConstraintFn,
) -> types.LogProbFn:
  """Build an unnormalized log density function."""
  del domains

  def target_log_prob_fn(state):
    logp = profile.log_prob(state)
    return jnp.where(constraint_fn(state), logp, -jnp.inf)

  return target_log_prob_fn
