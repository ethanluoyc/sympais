"""SYMPAIS algorithm."""
from typing import Optional

import jax
import jax.numpy as jnp

from sympais import constraint
from sympais import logger as logger_lib
from sympais import tasks
from sympais import z3_solver
from sympais.infer import importance
from sympais.infer import utils
from sympais.initializer import Initializer
from sympais.methods.importance import ProposalBuilder
from sympais.methods.importance import RandomWalkMetropolisKernel
from sympais.methods.importance import refine_domains
from sympais.methods.importance import sample_chain
from sympais.methods.importance import WindowedScaleAdaptor


def sympais(task: tasks.Task,
            seed: int,
            num_samples: int = int(1e6),
            num_proposals: int = 100,
            num_samples_per_iter: int = 5,
            proposal_scale_multiplier: float = 0.5,
            rmh_scale: float = 1.0,
            init: str = "z3",
            tune: bool = True,
            num_warmup_steps: int = 500,
            window_size: int = 100,
            resample: bool = True,
            proposal_std_num_samples: int = 100,
            logger: Optional[logger_lib.Logger] = None):
  """Run SYMPAIS with the RMH kernel.

  Refer to the inline comments for details on how to implement SYMPAIS given .
  different components.

  """
  profile = task.profile
  pcs = task.constraints
  # Build callable for 1_{PC}(x)
  constraint_fn = constraint.build_constraint_fn(pcs, task.domains)
  # Build the unnormalized density \bar{p}(x) = 1_PC(x) p(x)
  target_log_prob_fn = constraint.build_target_log_prob_fn(task.profile, task.domains,
                                                           constraint_fn)
  domains = task.domains
  key = jax.random.PRNGKey(seed)
  # The initializer finds initial feasible solution to bootstrap
  # the MCMC chains
  key, subkey = jax.random.split(key)
  initializer_ = Initializer(profile, pcs, domains, init, resample)
  try:
    initial_chain_state = initializer_(num_proposals, subkey)
  except z3_solver.UnsatConstraint:
    print("Unsatisfiable constraint")
    return {"mean": 0., "status": "unsat"}
  # Find a coarse approximation of the solution space.
  # This is used both by the RMH kernel for making proposals and
  # by the IS proposal for proposing from truncated distributions.
  refined_domains = refine_domains(pcs, domains)

  key, subkey = jax.random.split(key)
  # The proposal builder is a callable that
  # constructs importance sampling proposal distribution q(x)
  # for performing MIS at every iteration of PIMAIS
  proposal_builder = ProposalBuilder(profile, refined_domains,
                                     proposal_scale_multiplier,
                                     proposal_std_num_samples, subkey)
  # Construct a RMH transition kernel
  kernel = RandomWalkMetropolisKernel(target_log_prob_fn,
                                      jnp.ones(num_proposals) * rmh_scale,
                                      refined_domains)
  kernel.step = jax.jit(kernel.step)
  key, subkey = jax.random.split(key)

  # Initialize kernel parameters and run warmup
  # and optional parameter adaptation
  params = kernel.init()
  key, subkey = jax.random.split(key)
  if num_warmup_steps < 1:
    print("Not running warmup")
    chain_state = initial_chain_state
  else:
    if tune:
      print("Tuning the kernel")
      params, chain_state, _ = WindowedScaleAdaptor(kernel,
                                                    window_size)(subkey, params,
                                                                 initial_chain_state,
                                                                 num_warmup_steps)
    else:
      print("Not tuning the kernel")
      chain_state, (_, extra) = sample_chain(kernel, params, subkey,
                                             initial_chain_state, num_warmup_steps)
  print("Finished warm-up")
  # Comput the number of iterations given total sampling budget
  num_samples_warmup = num_proposals * num_warmup_steps
  num_iterations = (
      # 1) subtract the samples used during warmup
      (num_samples - num_samples_warmup)
      # 2) For each PI-MAIS iteration, we sample from each mixture component
      # `num_samples_per_iter` samples, plus an additional sample used by
      # each chain for making a single step of transition
      // ((num_proposals + 1) * num_samples_per_iter))

  # Initialize the state for running PI-MAIS.
  state = importance.pimais_init(chain_state)

  @jax.jit
  def pimais_step_fn(params, rng, state):
    kernel_fn = lambda key, state: kernel.step(params, key, state)
    return importance.pimais_step(rng, target_log_prob_fn, kernel_fn, proposal_builder,
                                  num_samples_per_iter, state)

  # Start running the PI-MAIS iterations
  rngs = jax.random.split(key, num_iterations)
  states = []
  extras = []
  for idx in range(num_iterations):
    # tic = time.time()
    state, extra = pimais_step_fn(params, rngs[idx], state)
    states.append(state)
    extras.append(extra)
    # Make sure async dispatch is accounted for in measuring running time.
    utils.block_until_ready((state, extra))
    # toc = time.time()
    # print("Time elapsed", toc - tic, "Mean", state.Ztot)
  if logger is not None:
    logger.write({"mean": state.Ztot})
  print("Final estimated probability {}".format(state.Ztot))
  # pylint would error on applying tree_multimap with *args. Silence it.
  # pylint: disable=no-value-for-parameter
  output = {
      "pimais_states": jax.tree_multimap(lambda *x: jnp.stack(x, 0), *states),
      "pimais_extras": jax.tree_multimap(lambda *x: jnp.stack(x, 0), *extras),
      "constraint_fn": constraint_fn,
      "target_log_prob_fn": target_log_prob_fn,
      "initial_chain_state": initial_chain_state,
      "proposal_builder": proposal_builder,
      "prob": state.Ztot,
  }
  return output
