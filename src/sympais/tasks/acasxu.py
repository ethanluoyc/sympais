"""Task for testing constraint satisfaction with ACAS Xu neural networks."""
import itertools
import os

import jax.numpy as jnp
import numpy as np
import sympy
from sympy import core

from sympais.distributions import TruncatedNormal

from . import base

try:
  from maraboupy import Marabou  # pytype: disable=import-error
except ImportError:
  Marabou = None

_NNET_FILE = ('../../../third_party/Marabou/src/input_parsers/acas_example/'
              'ACASXU_run2a_1_1_tiny.nnet')


def _iter_paths(network):
  relu_constraints = itertools.product(
      *[((c, False), (c, True)) for c in network.reluList])
  for c in relu_constraints:
    yield c


def _evaluate(network, input_values):
  # TODO consider non-normalized inputs and outputs
  env = {}
  for var, val in zip(network.inputVars.flatten().tolist(), input_values):
    env[var] = val
  for eqn in network.equList:
    # First variable corresponds the output variable
    outvar = eqn.addendList[0][1]
    # Compute the value of the outvar
    out = 0.
    for c, x in eqn.addendList[1:]:
      out += jnp.array(c) * env[x]
    out -= jnp.array(eqn.scalar)
    env[outvar] = out
    for rin, rout in network.reluList:
      if outvar == rin:
        env[rout] = jnp.maximum(0, env[rin])
  output = []
  for ov in network.outputVars[0].tolist():
    output.append(env[ov])
  return np.array(output), env


def test_constraint(env, pcs):
  out = True
  for pc in pcs:
    #         print(env[pc[0][1]], env[pc[0][0]], pc[1])
    out = out * ((env[pc[0][1]] == env[pc[0][0]]) == pc[1])
  return out


def print_path_constraints(pcs):
  for pc in pcs:
    print('{} == {}'.format(pc[0][1], pc[1]))


def generate_symbolic_constraints(network, pcs):
  store = {}
  for v in range(network.inputVars[0].size):
    store['x{}'.format(v)] = core.Symbol('x{}'.format(v))

  def eval_eqn(eqn):
    out = 0.
    for c, x in eqn.addendList[1:]:
      out = out + store['x{}'.format(x)] * c
    out = out - float(eqn.scalar)
    store['x{}'.format(eqn.addendList[0][1])] = out

  for eqn in network.equList:
    eval_eqn(eqn)
    for pc in pcs:
      invar, outvar = pc[0]
      if 'x{}'.format(outvar) not in store and 'x{}'.format(invar) in store:
        store['x{}'.format(outvar)] = sympy.Max(store['x{}'.format(invar)], 0)
  constraints = []
  for pc in pcs:
    invar, outvar = pc[0]
    if pc[1]:
      constraints.append(store['x{}'.format(invar)] >= 0)
    else:
      constraints.append(store['x{}'.format(invar)] <= 0)
  return constraints


class AcasXu(base.Task):
  """Task for testing constraint satisfaction with ACAS Xu neural networks.

  To use this task. You need to install `maraboupy`. Refer to

    https://neuralnetworkverification.github.io/Marabou/

  for instructions on how to install `maraboupy`. This repo includes
  a submodule which pins the version of the `maraboupy` used to produce the
  benchmark results.

  """

  def __init__(self, nnet_file: str = None, path_index: int = 0):
    profile = {
        'x0': TruncatedNormal(low=-100., high=100., loc=0, scale=1.),
        'x1': TruncatedNormal(low=-100., high=100., loc=0, scale=1.),
        'x2': TruncatedNormal(low=-100., high=100., loc=0, scale=1.),
        'x3': TruncatedNormal(low=-100., high=100., loc=0, scale=1.),
        'x4': TruncatedNormal(low=-100., high=100., loc=0, scale=1.),
    }
    domains = {}
    for k in profile:
      domains[k] = (-100., 100.)
    # Load the network from NNet file
    if nnet_file is None:
      here = os.path.dirname(os.path.abspath(__file__))
      nnet_file = os.path.abspath(os.path.join(here, _NNET_FILE))
    if Marabou is None:
      raise ImportError('maraboupy cannot be imported. You need '
                        'to install maraboupy to run the AcasXu benchmark.')
    net = Marabou.read_nnet(nnet_file)
    pcs_all = list(_iter_paths(net))
    pcs = pcs_all[path_index]
    constraints = generate_symbolic_constraints(net, pcs)
    super().__init__(profile, constraints, domains)
