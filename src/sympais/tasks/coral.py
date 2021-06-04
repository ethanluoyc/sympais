"""Coral benchmark tasks."""
# pylint: disable=line-too-long
# flake8: noqa
import glob
import importlib
import os
import re
import tempfile
from typing import List, Tuple

from sympais.tasks import base

SKIPPED_FILES = ["turn.nodeq.m", "conflict.nodeq.m"]


def compile_mathematica_example_to_python(source_file, target_file, path_index):
  # Take subjects from 'normal', which uses Gaussian distributions.
  # To use other distributions, change the distribution_pattern at line 29
  # path = 'subjects/normal/apollo.nodeq.m'
  path = source_file

  with open(path, "r") as mathematica_input_file:
    path_conditions = mathematica_input_file.readlines()

  mathematica_input = path_conditions[path_index]

  command_prefix = "TimeConstrained[ScientificForm[NProbability["
  command_suffix = '}],NumberFormat -> (#1 <> "E" <> #3 &)],1800] //AbsoluteTiming'

  mathematica_input = mathematica_input[len(command_prefix):-len(command_suffix) - 1]
  conjunctive_constraint, distributions = mathematica_input.split(
      ",{")  # Example: x1 < 10 && (x2 -1) + x3 >= -10;

  def translate_operators(mathematica_string: str) -> str:
    # Add any other replace rule if needed
    result = mathematica_string.replace("^", "**")
    return result

  # Extract individual conjuncts in a list
  conjuncts = [
      translate_operators(c.strip()) for c in conjunctive_constraint.split("&&")
  ]

  distribution_pattern = r"(?P<var_id>[a-zA-Z0-9_]+) \\\[Distributed\] TruncatedDistribution\[\{(?P<lb>-?[0-9]+(.[0-9]*)?),(?P<ub>-?[0-9]+(.[0-9]*)?)\},NormalDistribution\[(?P<loc>-?[0-9]+(.[0-9]*)?),(?P<stdev>-?[0-9]+(.[0-9]*)?)\]\]"
  matcher = re.compile(distribution_pattern)

  def translate_profile(
      mathematica_var_declaration: re.Match,) -> Tuple[str, str, List[str], str]:
    var_id = mathematica_var_declaration.group("var_id")
    lower_bound = float(mathematica_var_declaration.group("lb"))
    upper_bound = float(mathematica_var_declaration.group("ub"))
    gaussian_loc = float(mathematica_var_declaration.group("loc"))
    gaussian_stdev = float(mathematica_var_declaration.group("stdev"))

    symbolic_var_declaration = f'{var_id} = sympy.Symbol("{var_id}")'
    profile_specification = (
        f'"{var_id}": dist.TruncatedNormal(low={lower_bound}, high={upper_bound}, loc={gaussian_loc}, scale={gaussian_stdev})'
    )
    domain_constraints = [
        f"{var_id} >= {lower_bound}",
        f"{var_id} <= {upper_bound}",
    ]
    #     domain = [float(lower_bound), float(upper_bound)]
    domain = f'"{var_id}": ({lower_bound}, {upper_bound})'
    return (
        symbolic_var_declaration,
        profile_specification,
        domain_constraints,
        domain,
    )

  python_vars_declaration = []
  python_profile_specs = []
  python_constraints = conjuncts
  python_domains = []
  for m in matcher.finditer(distributions):
    (
        symbolic_var_declaration,
        profile_specification,
        domain_constraints,
        domain,
    ) = translate_profile(m)
    python_vars_declaration.append(symbolic_var_declaration)
    python_profile_specs.append(profile_specification)
    python_constraints += domain_constraints
    python_domains.append(domain)

  preamble = [
      "from sympy import Symbol",
      "import sympy",
      "from sympais import distributions as dist",
  ]
  python_code = ("\n".join(preamble) + "\n\n" + "\n".join(python_vars_declaration)
                 + "\n\nprofile = {\n\t" + ",\n\t".join(python_profile_specs)
                 + "\n}\n\n" + "constraints = [\n\t" + ",\n\t".join(python_constraints)
                 + "\n]\n" + "\n\ndomains = {\n\t" + ",\n\t".join(python_domains)
                 + "\n}\n\n")
  with open(target_file, "wt") as outfile:
    outfile.write(python_code)


def compile_mathematica_to_python_module(path, path_index):
  with tempfile.NamedTemporaryFile(mode="wt", suffix=".py") as tmp:
    filename = tmp.name
    compile_mathematica_example_to_python(path, filename, path_index)
    spec = importlib.util.spec_from_file_location("mod", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pytype: disable=attribute-error
    return module


def get_num_paths(filename):
  with open(filename, "r") as mathematica_input_file:
    path_conditions = mathematica_input_file.readlines()
  return len(path_conditions)


def list_all_tasks():
  files = glob.glob(
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "subjects/normal/*.m"))
  return [os.path.basename(file) for file in files]


class Coral(base.Task):

  def __init__(self, filename, path_index=0):
    if not os.path.exists(filename):
      filename = os.path.join(
          os.path.dirname(os.path.abspath(__file__)), "subjects/normal", filename)
    module = compile_mathematica_to_python_module(filename, path_index)
    self.num_paths = get_num_paths(filename)
    super().__init__(module.profile, module.constraints, module.domains)

  @property
  def num_dimensions(self):
    return len(self.profile)
