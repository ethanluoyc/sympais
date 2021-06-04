"""Classes for logging results."""
import json

import jax.numpy as jnp
import numpy as onp


def default(obj):
  if isinstance(obj, (jnp.ndarray, onp.ndarray)):
    return onp.array(obj).tolist()
  raise TypeError('Unknown type:', type(obj))


class Logger:
  """Base class for logging."""

  def write(self, metrics):
    raise NotImplementedError

  def close(self):
    raise NotImplementedError


class JsonLogger(Logger):
  """Simple logger that logs metrics as JSON lines."""

  def __init__(self, filename):
    super().__init__()
    self._filename = filename
    self._file = open(filename, 'wt')  # pylint: disable=consider-using-with

  def write(self, metrics):
    self._file.write(json.dumps(metrics, default=default))
    self._file.write('\n')
    self._file.flush()

  def close(self):
    self._file.close()


class InMemoryLogger(Logger):
  """Simple logger that logs metrics as JSON lines."""

  def __init__(self):
    super().__init__()
    self._results = []

  def write(self, metrics):
    self._results.append(metrics)
