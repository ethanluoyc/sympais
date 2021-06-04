import sympy

from sympais.experimental import smt


def test_example_smt_transformation():
  smt01 = '''
    (declare-fun x () Real)
    (declare-fun y () Real)

    (declare-const PI Real)

    (assert
    (<=
        (+
            (* (sin x) (sin (/ PI 2.0)))
            (* y (sqrt y))
        )
        1
        )
    )
    (assert
        (> x 20)
    )
    '''
  x = sympy.Symbol('x')
  y = sympy.Symbol('y')
  expected_result = (y * sympy.sqrt(y) + sympy.sin(x) <= 1) & (x > 20)
  assert sympy.logic.boolalg.Equivalent(expected_result,
                                        smt.smtlib_to_sympy_constraint(smt01))
