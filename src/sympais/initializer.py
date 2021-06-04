"""Initializer for bootstraping initial states."""
from typing import Optional

from absl import logging
import jax
from jax import random
import jax.numpy as jnp
import jax.scipy.special as jsp

from sympais import constraint
from sympais import profiles
from sympais import realpaver_solver
from sympais import types
from sympais import z3_solver


class Initializer:
  """Initializer for finding feasible initial solutions for bootstraping MCMC chains.

  This implements the solver used in SYMPAIS to find initial feasible solution.
  """

  def __init__(
      self,
      profile: profiles.Profile,
      constraints: types.Constraints,
      domains: types.DomainMap,
      init_strategy: str,
      resample: bool,
  ):
    """Create a initializer for finding initial feasible solutions.
      Args:
        profile: input `Profile` for the probabilistic program.
        constraints: input `Constraint` for the program analysis query.
        domains: input `Domain` for the input profile.
        init_strategy: "z3" or "realpaver", strategy used for finding initial solution.
        resample: if True, then the `num_solutions` of initial solution are found by
          re-sampling the set of candidate solutions via importance sampling (IS).
          Otherwise, the candidate solutions will be chosen uniformly.
      Raises:
        ValueError: if init_strategy is not in ("z3", "realpaver").
    """
    if init_strategy not in ("z3", "realpaver"):
      raise ValueError(f"Unknown initialization method {init_strategy}")
    self._profile = profile
    self._constraints = constraints
    self._domains = domains
    self._init = init_strategy
    self._resample = resample
    constraint_fn = jax.jit(constraint.build_constraint_fn(constraints, domains))
    self._target_log_prob_fn = jax.jit(
        constraint.build_target_log_prob_fn(profile, domains, constraint_fn))

  @staticmethod
  def _find_feasible_solution_realpaver(constraints: types.Constraints,
                                        domains: Optional[types.DomainMap] = None):
    constraint_fn = constraint.build_constraint_fn(constraints, domains)
    # First find interval boxes using RealPaver
    boxes = realpaver_solver.RealPaverSolver().solve(
        constraints, domains=domains, precision=0.01)
    solutions = []
    # All centers of the inner box would satisfy the constraints
    for box in list(boxes.inner_boxes):
      values = {
          site: interval.lower + (interval.upper - interval.lower) / 2
          for site, interval in box.intervals.items()
      }
      solutions.append(values)
    # Here we use the heuristic to find feasible solution
    # by computing the geometric center of each interval box and test for
    # constraint satisfaction.
    for box in list(boxes.outer_boxes):
      values = {
          site: interval.lower + (interval.upper - interval.lower) / 2
          for site, interval in box.intervals.items()
      }
      if constraint_fn(values):
        solutions.append(values)
    return solutions

  def __call__(self, num_solutions: int, rng: types.PRNGKey):
    """Find initial feasible solutions for the MCMC chains.
    Args:
      num_solutions: the number of solutions to be returned.
      rng: random seed to use
    """
    if self._init == "z3":
      states = [z3_solver.Z3Solver().solve(self._constraints, self._domains)]
      proposal_indices = random.randint(rng, (num_solutions,), minval=0, maxval=1)
      states_selected = []
      for pi in proposal_indices:
        states_selected.append(states[pi])
      proposal_states = (
          jax.tree_multimap(  # pylint: disable=no-value-for-parameter
              lambda *xs: jnp.stack(xs), *states_selected))
    elif self._init == "realpaver":
      feasible_states = self._find_feasible_solution_realpaver(
          self._constraints, self._domains)
      # Also add z3 feasible states. This guards against failures of not
      # getting a single feasible solution from RP.
      z3_feasible_states = [
          z3_solver.Z3Solver().solve(
              self._constraints,
              self._domains,
          )
      ]
      feasible_states.extend(z3_feasible_states)
      logging.info("Number of feasible solutions found is %d.", len(feasible_states))
      if len(feasible_states) < num_solutions:
        # First ensure that all feasible states are selected at least once.
        states_selected = []
        for state in feasible_states:
          states_selected.append(state)
        # Populate the rest of the states by sampling from feasible solutions
        if self._resample:
          logp = jnp.array(
              [self._target_log_prob_fn(state) for state in feasible_states])
          logp = logp - jsp.logsumexp(logp)
        else:
          logp = jnp.ones((len(feasible_states),)) / len(feasible_states)
        logging.debug("Initial state probs %s", jnp.exp(logp))
        proposal_indices = random.choice(
            rng,
            len(feasible_states),
            shape=(num_solutions - len(feasible_states),),
            replace=True,
            p=jnp.exp(logp),
        )
        for pi in proposal_indices:
          states_selected.append(feasible_states[pi])
      else:
        if self._resample:
          logp = jnp.array(
              [self._target_log_prob_fn(state) for state in feasible_states])
          logp = logp - jsp.logsumexp(logp)
        else:
          logp = jnp.ones((len(feasible_states),)) / len(feasible_states)
        logging.debug("Initial state probs %s", jnp.exp(logp))
        proposal_indices = random.choice(
            rng,
            len(feasible_states),
            shape=(num_solutions,),
            replace=True,
            p=jnp.exp(logp),
        )
        states_selected = []
        for pi in proposal_indices:
          states_selected.append(feasible_states[pi])
      proposal_states = (
          jax.tree_multimap(  # pylint: disable=no-value-for-parameter
              lambda *xs: jnp.stack(xs), *states_selected))
    else:
      raise ValueError("Unknown initialization strategy: {}".format(self._init))
    return proposal_states
