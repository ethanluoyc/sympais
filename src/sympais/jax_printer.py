"""Print SymPy expressions as JAX NumPy code.

NOTE(yl): Consider contributing this upstream, similar to

  https://github.com/sympy/sympy/pull/20516

  The code is adapted from the NumPy printer in SymPy.
  It's likely that JAX supports only a subset of NumPy, so
  we need to figure out which subset that is. Furthermore, JAX provides
  functions for SciPy, and we should ideally support those functions
  with the JaxPrinter as well.
"""
from typing import Iterable

from sympy import lambdify
# from sympy.abc import O
from sympy.core import S
from sympy.printing.codeprinter import CodePrinter
# from sympy.printing.pycode import _unpack_integral_limits
from sympy.printing.pycode import _known_functions_math
from sympy.printing.pycode import _print_known_const
from sympy.printing.pycode import _print_known_func
from sympy.printing.pycode import PythonCodePrinter


def _register_jax_lambdify():
  """Register JAX module to sympy.lambdify"""
  # pylint: disable=import-outside-toplevel
  from sympy.utilities.lambdify import MODULES

  if "JAX" in MODULES:
    return
  JAX = {"I": 1j}
  JAX_DEFAULT = JAX
  JAX_TRANSLATIONS = {}
  MODULES["jax"] = (JAX, JAX_DEFAULT, JAX_TRANSLATIONS, ("import jax;",))


_register_jax_lambdify()


def jaxify(args: Iterable, expr):
  """`jaxify` a SymPy expression."""
  # The settings passed to the printer is different the ones
  # used by default in lambdify. We should aim for the same defaults when
  # contributing this upstream.
  printer = JaxPrinter({
      "fully_qualified_modules": False,
      "inline": True,
      "allow_unknown_functions": False,
  })
  return lambdify(args, expr, modules="jax", printer=printer)


_not_in_numpy = "erf erfc factorial gamma loggamma".split()
_in_numpy = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_numpy]
_known_functions_numpy = dict(
    _in_numpy,
    **{
        "acos": "arccos",
        "acosh": "arccosh",
        "asin": "arcsin",
        "asinh": "arcsinh",
        "atan": "arctan",
        "atan2": "arctan2",
        "atanh": "arctanh",
        "exp2": "exp2",
        "sign": "sign",
        "logaddexp": "logaddexp",
        "logaddexp2": "logaddexp2",
    },
)
_known_constants_numpy = {
    "Exp1": "e",
    "Pi": "pi",
    "EulerGamma": "euler_gamma",
    "NaN": "nan",
    "Infinity": "PINF",
    "NegativeInfinity": "NINF",
}

_numpy_known_functions = {
    k: "jax.numpy." + v for k, v in _known_functions_numpy.items()
}
_numpy_known_constants = {
    k: "jax.numpy." + v for k, v in _known_constants_numpy.items()
}


class JaxPrinter(PythonCodePrinter):
  """
    Numpy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """

  _module = "jax"
  _kf = _numpy_known_functions
  _kc = _numpy_known_constants

  def __init__(self, settings=None):
    """
        `settings` is passed to CodePrinter.__init__()
        `module` specifies the array module to use, currently 'NumPy' or 'CuPy'
        """
    self.language = "Python with {}".format(self._module)
    self.printmethod = "_{}code".format(self._module)

    self._kf = {**PythonCodePrinter._kf, **self._kf}

    super().__init__(settings=settings)

  def _print_seq(self, seq):
    "General sequence printer: converts to tuple"
    # Print tuples here instead of lists because numba supports
    #     tuples in nopython mode.
    delimiter = ", "
    return "({},)".format(delimiter.join(self._print(item) for item in seq))

  def _print_MatMul(self, expr):
    "Matrix multiplication printer"
    if expr.as_coeff_matrices()[0] is not S.One:
      expr_list = expr.as_coeff_matrices()[1] + [(expr.as_coeff_matrices()[0])]
      return "({})".format(").dot(".join(self._print(i) for i in expr_list))
    return "({})".format(").dot(".join(self._print(i) for i in expr.args))

  def _print_MatPow(self, expr):
    "Matrix power printer"
    return "{}({}, {})".format(
        self._module_format(self._module + "numpy.linalg.matrix_power"),
        self._print(expr.args[0]),
        self._print(expr.args[1]),
    )

  def _print_Inverse(self, expr):
    "Matrix inverse printer"
    return "{}({})".format(
        self._module_format(self._module + "numpy.linalg.inv"),
        self._print(expr.args[0]),
    )

  def _print_DotProduct(self, expr):
    # DotProduct allows any shape order, but numpy.dot does matrix
    # multiplication, so we have to make sure it gets 1 x n by n x 1.
    arg1, arg2 = expr.args
    if arg1.shape[0] != 1:
      arg1 = arg1.T
    if arg2.shape[1] != 1:
      arg2 = arg2.T

    return "%s(%s, %s)" % (
        self._module_format(self._module + "numpy.dot"),
        self._print(arg1),
        self._print(arg2),
    )

  def _print_MatrixSolve(self, expr):
    return "%s(%s, %s)" % (
        self._module_format(self._module + "numpy.linalg.solve"),
        self._print(expr.matrix),
        self._print(expr.vector),
    )

  def _print_ZeroMatrix(self, expr):
    return "{}({})".format(
        self._module_format(self._module + "numpy.zeros"), self._print(expr.shape))

  def _print_OneMatrix(self, expr):
    return "{}({})".format(
        self._module_format(self._module + "numpy.ones"), self._print(expr.shape))

  def _print_FunctionMatrix(self, expr):
    # pylint: disable=import-outside-toplevel
    from sympy.abc import i
    from sympy.abc import j
    from sympy.core.function import Lambda

    lamda = expr.lamda
    if not isinstance(lamda, Lambda):
      lamda = Lambda((i, j), lamda(i, j))
    return "{}(lambda {}: {}, {})".format(
        self._module_format(self._module + ".numpy.fromfunction"),
        ", ".join(self._print(arg) for arg in lamda.args[0]),
        self._print(lamda.args[1]),
        self._print(expr.shape),
    )

  def _print_HadamardProduct(self, expr):
    func = self._module_format(self._module + ".numpy.multiply")
    return "".join("{}({}, ".format(func, self._print(arg))
                   for arg in expr.args[:-1]) + "{}{}".format(
                       self._print(expr.args[-1]), ")" * (len(expr.args) - 1))

  def _print_KroneckerProduct(self, expr):
    func = self._module_format(self._module + ".numpy.kron")
    return "".join("{}({}, ".format(func, self._print(arg))
                   for arg in expr.args[:-1]) + "{}{}".format(
                       self._print(expr.args[-1]), ")" * (len(expr.args) - 1))

  def _print_Adjoint(self, expr):
    return "{}({}({}))".format(
        self._module_format(self._module + ".numpy.conjugate"),
        self._module_format(self._module + ".numpy.transpose"),
        self._print(expr.args[0]),
    )

  def _print_DiagonalOf(self, expr):
    vect = "{}({})".format(
        self._module_format(self._module + ".numpy.diag"), self._print(expr.arg))
    return "{}({}, (-1, 1))".format(
        self._module_format(self._module + ".numpy.reshape"), vect)

  def _print_DiagMatrix(self, expr):
    return "{}({})".format(
        self._module_format(self._module + ".numpy.diagflat"),
        self._print(expr.args[0]),
    )

  def _print_DiagonalMatrix(self, expr):
    return "{}({}, {}({}, {}))".format(
        self._module_format(self._module + ".numpy.multiply"),
        self._print(expr.arg),
        self._module_format(self._module + ".numpy.eye"),
        self._print(expr.shape[0]),
        self._print(expr.shape[1]),
    )

  def _print_Piecewise(self, expr):
    "Piecewise function printer"
    exprs = "[{}]".format(",".join(self._print(arg.expr) for arg in expr.args))
    conds = "[{}]".format(",".join(self._print(arg.cond) for arg in expr.args))
    # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
    #     it will behave the same as passing the 'default' kwarg to select()
    #     *as long as* it is the last element in expr.args.
    # If this is not the case, it may be triggered prematurely.
    return "{}({}, {}, default={})".format(
        self._module_format(self._module + ".numpy.select"),
        conds,
        exprs,
        self._print(S.NaN),
    )

  def _print_Relational(self, expr):
    "Relational printer for Equality and Unequality"
    op = {
        "==": "equal",
        "!=": "not_equal",
        "<": "less",
        "<=": "less_equal",
        ">": "greater",
        ">=": "greater_equal",
    }
    if expr.rel_op in op:
      lhs = self._print(expr.lhs)
      rhs = self._print(expr.rhs)
      return "{op}({lhs}, {rhs})".format(
          op=self._module_format(self._module + ".numpy." + op[expr.rel_op]),
          lhs=lhs,
          rhs=rhs,
      )
    return super()._print_Relational(expr)

  def _print_And(self, expr):
    "Logical And printer"
    # We have to override LambdaPrinter because it uses Python 'and' keyword.
    # If LambdaPrinter didn't define it, we could use StrPrinter's
    # version of the function and add 'logical_and' to NUMPY_TRANSLATIONS.
    return self._expand_fold_binary_op(
        self._module_format(self._module + ".numpy.logical_and"), expr.args)

  def _print_Or(self, expr):
    "Logical Or printer"
    # We have to override LambdaPrinter because it uses Python 'or' keyword.
    # If LambdaPrinter didn't define it, we could use StrPrinter's
    # version of the function and add 'logical_or' to NUMPY_TRANSLATIONS.
    return self._expand_fold_binary_op(
        self._module_format(self._module + ".numpy.logical_or"), expr.args)

  def _print_Not(self, expr):
    "Logical Not printer"
    # We have to override LambdaPrinter because it uses Python 'not' keyword.
    # If LambdaPrinter didn't define it, we would still have to define our
    #     own because StrPrinter doesn't define it.
    return "{}({})".format(
        self._module_format(self._module + ".numpy.logical_not"),
        ",".join(self._print(i) for i in expr.args),
    )

  def _print_Pow(self, expr, rational=False):
    # pylint: disable=import-outside-toplevel
    # XXX Workaround for negative integer power error
    from sympy.core.power import Pow

    if expr.exp.is_integer and expr.exp.is_negative:
      expr = Pow(expr.base, expr.exp.evalf(), evaluate=False)
    return self._hprint_Pow(expr, rational=rational, sqrt=self._module + ".numpy.sqrt")

  def _print_Min(self, expr):
    return "{}(({}), axis=0)".format(
        self._module_format(self._module + ".numpy.amin"),
        ",".join(self._print(i) for i in expr.args),
    )

  def _print_Max(self, expr):
    return "{}(({}), axis=0)".format(
        self._module_format(self._module + ".numpy.amax"),
        ",".join(self._print(i) for i in expr.args),
    )

  def _print_arg(self, expr):
    return "%s(%s)" % (
        self._module_format(self._module + ".numpy.angle"),
        self._print(expr.args[0]),
    )

  def _print_im(self, expr):
    return "%s(%s)" % (
        self._module_format(self._module + ".numpy.imag"),
        self._print(expr.args[0]),
    )

  def _print_Mod(self, expr):
    return "%s(%s)" % (
        self._module_format(self._module + ".numpy.mod"),
        ", ".join(map(lambda arg: self._print(arg), expr.args)),
    )

  def _print_re(self, expr):
    return "%s(%s)" % (
        self._module_format(self._module + ".numpy.real"),
        self._print(expr.args[0]),
    )

  def _print_sinc(self, expr):
    return "%s(%s)" % (
        self._module_format(self._module + ".numpy.sinc"),
        self._print(expr.args[0] / S.Pi),
    )

  def _print_MatrixBase(self, expr):
    func = self.known_functions.get(expr.__class__.__name__, None)
    if func is None:
      func = self._module_format(self._module + ".numpy.array")
    return "%s(%s)" % (func, self._print(expr.tolist()))

  def _print_Identity(self, expr):
    shape = expr.shape
    if all([dim.is_Integer for dim in shape]):  # pylint: disable=R1729
      return "%s(%s)" % (
          self._module_format(self._module + ".numpy.eye"),
          self._print(expr.shape[0]),
      )
    else:
      raise NotImplementedError(
          "Symbolic matrix dimensions are not yet supported for identity matrices")

  def _print_BlockMatrix(self, expr):
    return "{}({})".format(
        self._module_format(self._module + ".numpy.block"),
        self._print(expr.args[0].tolist()),
    )

  def _print_ArrayTensorProduct(self, expr):
    array_list = [
        j for i, arg in enumerate(expr.args)
        for j in (self._print(arg), "[%i, %i]" % (2 * i, 2 * i + 1))
    ]
    return "%s(%s)" % (
        self._module_format(self._module + ".numpy.einsum"),
        ", ".join(array_list),
    )

  def _print_ArrayContraction(self, expr):
    # pylint: disable=import-outside-toplevel
    from sympy.tensor.array.expressions.array_expressions import \
        ArrayTensorProduct

    base = expr.expr
    contraction_indices = expr.contraction_indices
    if not contraction_indices:
      return self._print(base)
    if isinstance(base, ArrayTensorProduct):
      counter = 0
      d = {j: min(i) for i in contraction_indices for j in i}
      indices = []
      for rank_arg in base.subranks:
        lindices = []
        for i in range(rank_arg):
          if counter in d:
            lindices.append(d[counter])
          else:
            lindices.append(counter)
          counter += 1
        indices.append(lindices)
      elems = [
          "%s, %s" % (self._print(arg), ind) for arg, ind in zip(base.args, indices)
      ]
      return "%s(%s)" % (
          self._module_format(self._module + ".numpy.einsum"),
          ", ".join(elems),
      )
    raise NotImplementedError()

  def _print_ArrayDiagonal(self, expr):
    diagonal_indices = list(expr.diagonal_indices)
    if len(diagonal_indices) > 1:
      # TODO: this should be handled in sympy.codegen.array_utils,
      # possibly by creating the possibility of unfolding the
      # ArrayDiagonal object into nested ones. Same reasoning for
      # the array contraction.
      raise NotImplementedError
    if len(diagonal_indices[0]) != 2:
      raise NotImplementedError
    return "%s(%s, 0, axis1=%s, axis2=%s)" % (
        self._module_format(self._moduel + ".numpy.diagonal"),
        self._print(expr.expr),
        diagonal_indices[0][0],
        diagonal_indices[0][1],
    )

  def _print_PermuteDims(self, expr):
    return "%s(%s, %s)" % (
        self._module_format(self._module + ".numpy.transpose"),
        self._print(expr.expr),
        self._print(expr.permutation.array_form),
    )

  def _print_ArrayAdd(self, expr):
    return self._expand_fold_binary_op(self._module + ".numpy.add", expr.args)

  # pylint: disable=protected-access
  _print_lowergamma = CodePrinter._print_not_supported
  _print_uppergamma = CodePrinter._print_not_supported
  _print_fresnelc = CodePrinter._print_not_supported
  _print_fresnels = CodePrinter._print_not_supported


for func_ in _numpy_known_functions:
  setattr(JaxPrinter, f"_print_{func_}", _print_known_func)

for const in _numpy_known_constants:
  setattr(JaxPrinter, f"_print_{const}", _print_known_const)
