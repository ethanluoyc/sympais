# Benchmark Tasks

This folder includes benchmark tasks from different domains.

## List of Implemented Tasks

| Task      | Description   | Code |
| ----------| -----------   | ---- |
| sphere  | Spheres with different dimensionality | [sphere.py](./sphere.py) |
| cube  | 2 dimensional Cube | [cube.py](./cube.py) |
| torus  | 3 Dimensional torus | [torus.py](./torus.py) |
| coral  | Coral microbenchmarks [1] | [coral.py](./coral.py) |
| vinci  | Vinci convex polytopes [2] | [vinci.py](./vinci.py) |
| acasxu  | ACAS Xu [3] NN constraints | [acasxu.py](./acasxu.py) |

## Notes

- The coral benchmarks are adapted from http://qcoral.s3-website-us-west-2.amazonaws.com.
- For ACAS Xu, since we use the NNFile parser.
from [Marabou](https://github.com/NeuralNetworkVerification/Marabou) [4],
you need to install Marabou before using these task definitions. Refer to Marabou's
documentation on how to install Marabou's Python interface `maraboupy`.
- The Vinci input files were obtained from http://www.multiprecision.org/vinci/polytopes.html.

## Defining New Tasks

You can create new task definitions by subclassing `base.Task`. Call the
`__init__` method of the parent by providing a profile for the program inputs,
a list of path constraints and the domains for the program inputs.

For example, consider a task with the program input `x` which follows a standard Gaussian distribution in the domain '\[-10,10\]'. We would like to compute the probability for the path constraint
x >= 0. To define such a task, we can write
```python
import sympy
from sympais import tasks
from numpyro import distributions

x = sympy.Symbol('x')

task tasks.Task({'x': distributions.Normal(0, 1)}, (x >= 0), {'x': (-10, 10)})
```

Notice that in the definition of the domains, we are using the string literal `'x'`
instead of `x`.

### Known limitation
Currently
- the list of constraints cannot contain conjunction (e.g. `sympy.And`, `sympy.Or`)
- Only continuous distribution is currently supported. Using discrete distribution will issue
a warning and the implementation is not tested. This is not a limitation of SYMPAIS
but a limitation of the implementation.

## References
[1] Mateus Borges, Antonio Filieri, Marcelo d'Amorim, Corina S. Păsăreanu, and Willem Visser. 2014.Compositional solution space quantification for probabilistic software analysis. SIGPLAN Not. 49, 6 (June 2014), 123–132. DOI:https://doi.org/10.1145/2666356.2594329

[2] Benno Büeler, Andreas Enge and Komei Fukuda: Exact Volume Computation for Polytopes: A Practical Study.

[3] M. P. Owen, A. Panken, R. Moss, L. Alvarez and C. Leeper, "ACAS Xu: Integrated Collision Avoidance and Detect and Avoid Capability for UAS," 2019 IEEE/AIAA 38th Digital Avionics Systems Conference (DASC), 2019, pp. 1-10, doi: 10.1109/DASC43569.2019.9081758.

[4] Katz, Guy, et al. "The marabou framework for verification and analysis of deep neural networks." International Conference on Computer Aided Verification. Springer, Cham, 2019.
