"""Stratified sampling for probabilistic program analysis."""
import time
from typing import Optional

from absl import logging
import jax.numpy as np
import numpy as onp
from scipy import stats

from sympais import constraint
from sympais import logger as logger_lib
from sympais import profiles
from sympais import tasks
from sympais.realpaver_solver import RealPaverSolver


def _compute_box_prob_mass(profile, b):
  area = 1
  for name, int_ in b.intervals.items():
    target_dist = profile[name]
    area *= target_dist.cdf(int_.upper) - target_dist.cdf(int_.lower)
  return area


def truncnorm(loc, scale, lower, upper, num_samples, random_state=None):
  a = (lower - loc) / scale
  b = (upper - loc) / scale
  return stats.truncnorm(
      loc=loc, scale=scale, a=a, b=b).rvs(
          num_samples, random_state=random_state)


def run_stratified(task: tasks.Task,
                   seed: int,
                   num_samples: int,
                   batch_size: Optional[int] = None,
                   box_precision: float = 0.1,
                   logger: Optional[logger_lib.Logger] = None):
  """Run the stratified sampler for probabilistic program analysis.

  This resembles a simplified implementation qCoral that uses interval
  boxes produced by RealPaver to perform stratified sampling. This is equivalent
  to the sampling scheme used by qCoral for analyzing the sliced constraints for
  a given path.
  """

  profile = task.profile
  assert isinstance(profile, profiles.IndependentProfile)
  profile = profile.components
  domains = task.domains

  # Compute the mass of the truncated cdf
  assert isinstance(profile, dict)

  pcs = task.constraints
  boxes = RealPaverSolver().solve(pcs, precision=box_precision, domains=task.domains)
  constraint_fn = constraint.build_constraint_fn(pcs, domains)

  outer_boxes = boxes.outer_boxes
  inner_boxes = boxes.inner_boxes

  path_inner_area = sum([_compute_box_prob_mass(profile, b) for b in inner_boxes])
  outer_box_area = [_compute_box_prob_mass(profile, b) for b in outer_boxes]
  path_tot_area = sum([_compute_box_prob_mass(profile, b) for b in boxes])
  path_outer_area = path_tot_area - path_inner_area

  logging.info("inner vol %f, total vol %f", path_inner_area, path_tot_area)

  def _run_batch(rng, num_samples):
    estimate = path_inner_area
    var = 0.0
    # Only sample the outer boxes
    outer_boxes = boxes.outer_boxes
    # Compute the number of samples allocated for each box
    assert len(num_samples) == len(outer_boxes)
    for box_idx, b in enumerate(outer_boxes):
      area = outer_box_area[box_idx]
      num_samples_box = num_samples[box_idx]
      if num_samples_box == 0:
        continue
      proposals = _sample_box(rng, b, num_samples_box)
      box_mean = np.mean(constraint_fn(proposals))
      estimate += area * box_mean
      var += np.square(area) * (1 - box_mean) * (box_mean) / num_samples_box
    return {"mean": estimate, "var": var}

  def _sample_box(rng, box, num_samples):
    proposals = {}
    for k in profile:
      mean = profile[k].loc
      std = profile[k].scale
      if k not in box.intervals:
        # Some intervals may not have a particular variable,
        # which implies that the variable is not constrained
        lower = domains[k][0]
        upper = domains[k][1]
      else:
        interval = box.intervals[k]
        lower = interval.lower
        upper = interval.upper
      proposals[k] = truncnorm(
          loc=mean,
          scale=std,
          lower=lower,
          upper=upper,
          num_samples=num_samples,
          random_state=rng)
    return proposals

  def _allocate_samples(budget):
    outer_boxes = boxes.outer_boxes
    total_area = path_outer_area
    num_samples_boxes = [0] * len(outer_boxes)
    remainders = [0] * len(outer_boxes)
    allocated = 0

    for n in range(len(outer_boxes)):
      area = outer_box_area[n]
      proportion = area / total_area
      quota = budget * proportion
      num_samples_boxes[n] = int(onp.floor(quota))
      remainders[n] = quota - num_samples_boxes[n]
      allocated += num_samples_boxes[n]
    # Allocate remainders
    if allocated < budget:
      to_allocate = budget - allocated
      sort_index = onp.argsort(onp.array(remainders))[::-1]
      sort_index = sort_index[:to_allocate]
      for i in sort_index:
        num_samples_boxes[i] += 1
    return num_samples_boxes

  rng = onp.random.RandomState(seed)
  if batch_size is None:
    batch_size = num_samples
  num_batches = num_samples // batch_size
  assert num_samples % batch_size == 0
  num_samples_per_path = batch_size
  path_samples_sizes = _allocate_samples(num_samples_per_path)

  start_time = time.time()
  mean = 0.0
  var = 0.0
  n = 0
  for batch_idx in range(num_batches):
    tic = time.time()
    batch_result = _run_batch(rng, path_samples_sizes)
    batch_mean = batch_result["mean"]
    batch_var = batch_result["var"]
    mean = (batch_size * batch_idx * mean + batch_mean * batch_size) / (
        (batch_idx + 1) * batch_size)
    var = (
        batch_size / (n + batch_size) * batch_var + n / (n + batch_size) * var +
        (batch_size * n / ((batch_size + n)**2) * np.square(mean - batch_mean)))
    n += batch_size
    toc = time.time()
    logging.info(
        "mean %f, batch mean %f, num_samples %d, elapsed %.4f seconds",
        mean,
        batch_mean,
        (batch_idx + 1) * batch_size,
        toc - tic,
    )
    logs = {
        "mean": mean,
        "var": var,
        "cov": np.sqrt(var) / mean,
        "sample_count": (batch_idx + 1) * batch_size,
        "batch_mean": batch_mean,
        "batch_var": batch_var,
        "time": time.time() - start_time
    }
    if logger is not None:
      logger.write(logs)
  logging.info("Finished sampling, time %f seconds", time.time() - start_time)
  # if logger is not None:
  #   logger.write({"mean": mean})
  return {"mean": mean, "var": var}
