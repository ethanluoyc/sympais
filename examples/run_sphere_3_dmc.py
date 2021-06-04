from absl import app
import jax

from sympais import tasks
from sympais.methods.dmc import run_dmc


def main(argv):
  del argv
  task = tasks.Sphere(3)
  run_dmc(task, seed=0, num_samples=int(1e8), batch_size=int(1e7))


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
