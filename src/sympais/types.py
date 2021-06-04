"""Core types used by SYMPAIS."""
from typing import Any, Callable, Dict, List, Tuple, Union

import jax.numpy as jnp
from sympy import core

NestedArray = Any  # Change to recursive type definition
Array = jnp.ndarray
PRNGKey = jnp.ndarray

Constraint = core.Expr
Constraints = Union[List[core.Expr], Tuple[core.Expr]]
ConstraintFn = Callable[[NestedArray], Array]

Domain = Tuple[float, float]
DomainMap = Dict[str, Domain]

LogProbFn = Callable[[NestedArray], Array]
