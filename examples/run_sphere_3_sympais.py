from absl import app
import jax

from sympais import tasks
from sympais.methods.importance import run_sympais


def main(argv):
  del argv
  task = tasks.Sphere(3)

  result = run_sympais(
      task=task,
      key=jax.random.PRNGKey(0),
      num_samples=int(1e6),
      num_proposals=100,
      num_samples_per_iter=5,
      proposal_scale_multiplier=0.5,
      rmh_scale=1.0,
      tune=True,
      init="realpaver",
      resample=True,
      num_warmup_steps=500,
      window_size=100,
  )
  print("Final result is ", float(result["mean"]))


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
