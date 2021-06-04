# Grammar adapted from https://github.com/marcpare/baltasar.git
# ---------------------------------------------------------
#
# Implement Realpaver constructs as Python objects.
#
# quick_constraint implemented using custom grammar
#
# ---------------------------------------------------------
# pylint: disable=all
# flake8: noqa

import numpy as np


class Constant:

  def __init__(self, var_name, value):
    self.var_name = var_name
    self.value = value

  def render(self):
    return "%s = %s" % (self.var_name, self.value)


def _maybe_convert_infinity(val):
  if np.isinf(val):
    if float(val) > 0:
      sign = "+"
    else:
      sign = "-"
    return sign + "oo"
  return val


class Variable:

  def __init__(self,
               var_name,
               lower_bound,
               upper_bound,
               defined_over=None,
               var_type=None):
    self.var_name = var_name
    self.lower_bound = _maybe_convert_infinity(lower_bound)
    self.upper_bound = _maybe_convert_infinity(upper_bound)
    self.defined_over = defined_over
    if var_type:
      self.var_name = "%s %s" % (var_type, var_name)

  def render(self):
    left_bracket = "["
    if self.lower_bound == "-oo":
      left_bracket = "]"
    right_bracket = "]"
    if self.upper_bound == "+oo":
      right_bracket = "["

    if self.defined_over:
      a = self.defined_over[0]
      b = self.defined_over[-1]
      return "%s[%s..%s] in %s%s, %s%s" % (self.var_name, a, b, left_bracket,
                                           self.lower_bound, self.upper_bound,
                                           right_bracket)
    else:
      return "%s in %s%s, %s%s" % (self.var_name, left_bracket, self.lower_bound,
                                   self.upper_bound, right_bracket)


class RealPaverInput:

  def __init__(self, timeout=None):
    self.constants = []
    self.variables = []
    self.variable_domain = {}
    self.constant_domain = {}
    self.constraints = []
    self.ast = make_parser()
    self.var_lookup = {}
    self.sets = {}
    self._timeout = timeout

  def add_constant(self, cname, cvalue, defined_over=None):
    self.constant_domain[cname] = defined_over
    # unroll constants defined over an interval
    if defined_over:
      for (x, y) in zip(defined_over, cvalue):
        self.constants.append(Constant("%s_%s" % (cname, x), y))
    else:
      self.constants.append(Constant(cname, cvalue))

  def add_variable(self,
                   var_name,
                   lower_bound,
                   upper_bound,
                   defined_over=None,
                   var_type=None):
    self.variable_domain[var_name] = defined_over
    self.variables.append(
        Variable(var_name, lower_bound, upper_bound, defined_over, var_type))

  def add_constraint(self, constraint):
    self.constraints.append(constraint)

  def render(self):
    ret = ""
    if len(self.constants) > 0:
      ret += "Constants\n"
      rendered_constants = " , \n".join(["  " + c.render() for c in self.constants])
      ret += rendered_constants + " ;"

    ret += "\n\n"
    ret += "Variables\n"
    rendered_variables = " , \n".join(["  " + v.render() for v in self.variables])
    ret += rendered_variables + " ;"

    ret += "\n\n"
    ret += "Constraints\n"
    rendered_constraints = " , \n".join(["  " + c for c in self.constraints])
    ret += rendered_constraints + " ;"

    if self._timeout is not None:
      ret += f"\n\nTime = {self._timeout} ;"
    return ret


def make_parser():
  import ply.lex as lex
  import ply.yacc as yacc

  # ---------------------------------
  # Tokenizing rules
  # ---------------------------------

  reserved = {'for': 'FOR', 'in': 'IN'}

  tokens = [
      'ID',
      'FLOAT',
      'INT',
      'PLUS',
      'MINUS',
      'TIMES',
      'DIVIDE',
      'EQUALS',
      'UMINUS',
      'UPLUS',
      'GREATER_THAN',
      'LESS_THAN',
      'LPAREN',
      'RPAREN',
      'LBRACK',
      'RBRACK',
      'COMMA',
  ] + list(reserved.values())

  # Tokens
  t_PLUS = r'\+'
  t_MINUS = r'-'
  t_TIMES = r'\*'
  t_DIVIDE = r'/'
  t_EQUALS = r'='
  t_LESS_THAN = r'<'
  t_GREATER_THAN = r'>'
  t_LPAREN = r'\('
  t_RPAREN = r'\)'
  t_LBRACK = r'\['
  t_RBRACK = r'\]'
  t_COMMA = r','

  def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')  # Check for reserved words
    return t

  t_ignore = " \t"

  def t_FLOAT(t):
    r'[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?'
    t.value = float(t.value)
    return t

  def t_INT(t):
    r'\d+'
    t.value = int(t.value)
    return t

  def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

  def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

  # Build the lexer
  lexer = lex.lex()

  # ---------------------------------
  # Calculator parsing rules
  # ---------------------------------

  precedence = (
      ('left', 'PLUS', 'MINUS'),
      ('left', 'TIMES', 'DIVIDE'),
      ('right', 'UMINUS', 'UPLUS'),
      ('left', 'EQUALS'),
      ('left', 'FOR'),
  )

  def p_statement_eqn(p):
    'statement : equation'
    p[0] = p[1]

  def p_expression_equation(p):
    '''equation : expression EQUALS expression
                    | expression LESS_THAN expression
                    | expression GREATER_THAN expression '''
    p[0] = ('equation-expression', p[2], p[1], p[3])

  def p_statement_for(p):
    '''statement : equation FOR ID IN ID '''
    p[0] = ('for-expression', p[3], p[5], p[1])

  def p_expression_arguments(p):
    '''arguments : expression COMMA arguments
                     | expression '''
    if len(p) == 2:
      p[0] = ('argument', p[1])
    else:
      p[0] = ('argument-list', p[1], p[3])

  def p_expression_func(p):
    '''expression : ID LPAREN arguments RPAREN '''
    p[0] = ('func-expression', p[1], p[3])

  def p_expression_binop(p):
    '''expression : expression PLUS expression
                      | expression MINUS expression
                      | expression TIMES expression
                      | expression DIVIDE expression'''
    # if p[2] == '+'  : p[0] = p[1] + p[3]
    p[0] = ('binary-expression', p[2], p[1], p[3])

  def p_expression_group(p):
    '''expression : LPAREN expression RPAREN '''
    p[0] = ('group-expression', p[2])

  def p_expression_int(p):
    'expression : INT'
    p[0] = ('int-expression', p[1])

  def p_expression_float(p):
    'expression : FLOAT'
    p[0] = ('float-expression', p[1])

  def p_expression_uminus(p):
    '''expression : MINUS expression %prec UMINUS
                      | PLUS expression %prec UPLUS '''
    p[0] = ('unary-expression', p[2], p[1])

  def p_expression_array(p):
    "expression : ID LBRACK expression RBRACK"
    p[0] = ('array-expression', p[1], p[3])

  def p_expression_id(p):
    "expression : ID"
    p[0] = ('variable-expression', p[1])

  def p_error(p):
    if p:
      print("Syntax error at '%s'" % p.value)
    else:
      print("Syntax error at EOF")

  # Build the parser
  parser = yacc.yacc()

  # ---------------------------------
  # Input function
  # ---------------------------------

  def input(text):
    result = parser.parse(text, lexer=lexer)
    return result

  return input

  def roll_up_arguments(self, node):
    if node[0] == 'argument-list':
      ret = [self.parse(node[1])]
      ret.extend(self.roll_up_arguments(node[2]))
      return ret
    else:
      return [self.parse(node[1])]

  def parse(self, node):
    # determine the type of node
    parse = self.parse
    if node[0] == 'binary-expression':
      if node[1] == '+':
        return "(%s + %s)" % (parse(node[2]), parse(node[3]))
      elif node[1] == '*':
        return "(%s * %s)" % (parse(node[2]), parse(node[3]))
      elif node[1] == '/':
        return "(%s / %s)" % (parse(node[2]), parse(node[3]))
      elif node[1] == '-':
        return "(%s - %s)" % (parse(node[2]), parse(node[3]))
    elif node[0] == 'unary-expression':
      if node[2] == '-':
        return "(-(%s))" % parse(node[1])
      elif node[2] == '+':
        return "(+(%s))" % parse(node[1])
    elif node[0] == 'float-expression' or node[0] == 'int-expression':
      return node[1]
    elif node[0] == 'variable-expression':
      if node[1] not in self.constant_domain and node[
          1] not in self.variable_domain and node[1] not in self.var_lookup:
        print("Did not add %s to the problem" % node[1])
      else:
        return self.var_lookup.get(node[1], node[1])
    elif node[0] == 'array-expression':
      index = parse(node[2])
      try:
        index = eval(index())
      except:
        pass
      if node[1] in self.constant_domain and self.constant_domain[node[1]] is not None:
        return "%s_%s" % (node[1], index)
      elif node[1] in self.variable_domain and self.variable_domain[
          node[1]] is not None:
        return "%s[%s]" % (node[1], eval(str(index)))
      elif node[1] in self.constant_domain and self.constant_domain[node[1]] is None:
        print("Constant %s referenced as a variable" % node[1])
      elif node[1] in self.variable_domain and self.variable_domain[node[1]] is None:
        print("Variable %s referenced as a constant" % node[1])
      else:
        print("Array expression neither a constant nor a variable.")

    elif node[0] == 'for-expression':
      over_set = self.sets[node[2]]
      ret = []
      for x in over_set:
        self.var_lookup[node[1]] = x
        ret.append(parse(node[3]))
      return ret
    elif node[0] == 'equation-expression':
      if node[1] == '=':
        return "%s = %s" % (parse(node[2]), parse(node[3]))
      elif node[1] == '<':
        return "%s <= %s" % (parse(node[2]), parse(node[3]))
      elif node[1] == '>':
        return "%s >= %s" % (parse(node[2]), parse(node[3]))
    elif node[0] == 'group-expression':
      return parse(node[1])
    elif node[0] == 'func-expression':
      # roll up argument list
      arguments = self.roll_up_arguments(node[2])

      # for now, functions hard-coded
      if node[1] == 'sum':
        domain = self.sets[node[2][2][1][1]]
        the_var = node[2][1][1]
        unrolled = []
        for x in domain:
          unrolled.append("%s[%s]" % (the_var, x))
        joined = " + ".join(unrolled)
        return "(%s)" % joined
      elif node[1] == 'min' or node[1] == 'max':
        return "%s(%s)" % (node[1], ",".join(map(str, arguments)))
      elif node[1] == 'tanh':
        return "%s(%s)" % (node[1], ",".join(map(str, arguments)))
      else:
        raise ("Unknown function %s" % node[1])

  def quick_constraint(self, raw_equation):
    foo = self.ast(raw_equation)
    parsed = self.parse(foo)

    if isinstance(parsed, list):
      for x in parsed:
        self.add_constraint(x)
    else:
      self.add_constraint(parsed)

    return parsed

  def add_set(self, set_name, set_items):
    self.sets[set_name] = set_items
