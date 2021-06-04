import shutil
import unittest

import jax

from sympais.methods.dmc import run_dmc
from sympais.methods.importance import run_sympais
from sympais.methods.importance import run_sympais_hmc
from sympais.methods.stratified import run_stratified
from sympais.tasks import Torus

REALPAVER_NOT_INSTALLED = shutil.which("realpaver") is None


class MethodsTest(unittest.TestCase):

  def test_dmc(self):
    task = Torus()
    result = run_dmc(task, seed=0, num_samples=int(1e5), batch_size=int(1e4))
    print("result", result)

  @unittest.skipIf(REALPAVER_NOT_INSTALLED, "No realpaver found")
  def test_sympais(self):
    task = Torus()
    run_sympais(
        task=task,
        key=jax.random.PRNGKey(0),
        num_proposals=10,
        num_samples_per_iter=5,
        proposal_scale_multiplier=0.5,
        rmh_scale=1.0,
        tune=True,
        num_warmup_steps=50,
        window_size=10,
    )

  @unittest.skipIf(REALPAVER_NOT_INSTALLED, "No realpaver found")
  def test_sympais_hmc(self):
    task = Torus()
    run_sympais_hmc(
        task=task,
        key=jax.random.PRNGKey(0),
        num_proposals=10,
        num_samples_per_iter=5,
        proposal_scale_multiplier=0.5,
    )

  @unittest.skipIf(REALPAVER_NOT_INSTALLED, "No realpaver found")
  def test_stratified(self):
    task = Torus()
    result = run_stratified(task, seed=0, num_samples=int(1e5), batch_size=int(1e4))
    print("strat", result)


if __name__ == "__main__":
  unittest.main()
