"""Direct Monte Carlo baseline"""
from typing import Optional

from absl import logging
from jax import random
import jax.numpy as np

from sympais import constraint
from sympais import logger as logger_lib
from sympais import tasks


def run_dmc(task: tasks.Task,
            seed: int,
            num_samples: int,
            batch_size: Optional[int] = None,
            logger: Optional[logger_lib.Logger] = None):
  """Run the Direct Monte Carlo (DMC) baseline.

  Args:
    task: `Task` for performing probabilistic program analysis.
    seed: random number seed.
    num_samples: Total number of samples used by DMC.
    batch_size: if not `None`, the DMC sampler is run with batches of samples.
    logger: if not None, log the results with the logger.

  Returns:
    A dict of the estimates found by the DMC sampler.

  """
  constraint_fn = constraint.build_constraint_fn(task.constraints, task.domains)
  rng = random.PRNGKey(seed)
  if batch_size is None:
    batch_size = num_samples
  num_batches = num_samples // batch_size
  assert num_samples % batch_size == 0

  mean = 0.
  var = 0.
  for batch_idx in range(num_batches):
    rng, subkey = random.split(rng)
    samples = task.profile.sample(subkey, sample_shape=(batch_size,))
    batch_mean = np.mean(constraint_fn(samples))
    # batch_var = batch_mean * (1 - batch_mean) / batch_size
    # mean_sum += batch_mean
    # mean_agg = mean_sum / (batch_idx + 1)
    mean = (batch_size * batch_idx * mean + batch_mean * batch_size) / (
        (batch_idx + 1) * batch_size)
    var = (1 - mean) * mean / ((batch_idx + 1) * batch_size)
    logs = {
        "mean": mean,
        "var": var,
        "batch_mean": batch_mean,
        "sample_count": (batch_idx + 1) * batch_size,
        # "time": time.time() - start_time,
        # "cov": np.sqrt(var) / mean
    }
    if logger is not None:
      logger.write(logs)
    # self._log(results)
    # results_all.append(results)
    logging.info("mean %s, var %s", mean, var)
  # if logger is not None:
  #   logger.write({"mean": mean})
  return {"mean": mean, "var": mean * (1 - mean) / num_samples}
