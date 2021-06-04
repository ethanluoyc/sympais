import os

from sympy import Symbol

import sympais.distributions as dist

from . import base


def parse_name(line):
  return line.strip()


def parse_spec(line):
  parts = line.strip().split()
  assert len(parts) == 3
  num_constraints = int(parts[0])
  num_variables = int(parts[1]) - 1
  num_type = parts[2]
  return num_constraints, num_variables, num_type


def parse_constraint(line):
  parts = line.strip().split()
  parts = list(map(eval, parts))  # Not safe!
  b = parts[0]
  A = list(map(lambda x: -x, parts[1:]))
  return A, b


def find_vinci_tasks():
  found_files = {}
  vinci_inputs_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "vinci_inputs")
  for root, _, files in os.walk(vinci_inputs_dir, topdown=False):
    for file in files:
      if file.endswith(".ine"):
        task_name = file.replace(".ine", "")
        assert task_name not in found_files
        found_files[task_name] = os.path.join(root, file)
  return found_files


def parse_constraint_info(lines, pos):
  assert lines[pos].startswith("begin")
  num_constraints, num_variables, num_type = parse_spec(lines[pos + 1])
  constraints = []
  for i in range(num_constraints):
    constraints.append(parse_constraint(lines[pos + i + 2]))
  assert lines[pos + 2 + num_constraints].startswith("end")
  return (
      (num_constraints, num_variables, num_type, constraints),
      pos + 2 + num_constraints,
  )


def parse_vinci_program(lines):
  i = 0
  while i < len(lines):
    line = lines[i]
    if line.startswith("begin"):
      body, i = parse_constraint_info(lines, i)
    else:
      i += 1
  (num_constraints, num_variables, num_type, constraints) = body
  return (num_constraints, num_variables, num_type, constraints)


def _get_constraints(filename):
  if not os.path.exists(filename):
    vinci_tasks = find_vinci_tasks()
    filename = vinci_tasks.get(
        filename,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vinci_inputs", filename))
    if not os.path.exists(filename):
      raise ValueError("Unable to find task {}".format(filename))

  with open(filename) as infile:
    lines = infile.readlines()
  num_constraints, num_variables, _, _constraints = parse_vinci_program(lines)
  i = 0
  variables = [Symbol("x{}".format(i)) for i in range(num_variables)]
  variable_names = ["x{}".format(i) for i in range(num_variables)]
  constraints = []
  for constraint in _constraints:
    A, b = constraint
    s = None
    for j in range(len(A)):  # pylint: disable=consider-using-enumerate
      if s is None:
        s = A[j] * variables[j]
      else:
        s += A[j] * variables[j]
    constraint = s <= b
    constraints.append(constraint)
    i += 1
  return variable_names, constraints, num_variables, num_constraints


class Vinci(base.Task):
  """Benchmark tasks for problems from Vinci."""

  def __init__(self, filename, loc=0.0, scale=1.0):
    variables, constraints, _, _ = _get_constraints(filename)
    profile = {
        v: dist.TruncatedNormal(low=-100., high=100., loc=loc, scale=scale)
        for v in variables
    }
    domains = {}
    for var in variables:
      domains[var] = (-100., 100.)
    super().__init__(profile, constraints, domains)


def get_vinci_info(filename):
  _, _, num_variables, num_constraints = _get_constraints(filename)
  return num_variables, num_constraints
