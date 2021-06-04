import logging
import os

import hydra
import jax
from omegaconf import OmegaConf

from sympais import methods
from sympais import tasks
from sympais.logger import JsonLogger

log = logging.getLogger(__name__)

_TASKS = {
    "torus": tasks.Torus,
    "coral": tasks.Coral,
    "sphere": tasks.Sphere,
    "vinci": tasks.Vinci,
    "cube2d": tasks.Cube2d,
    "acasxu": tasks.AcasXu,
}


def _build_task(task_config):
  config = task_config.copy()
  task_name = config.pop("name")
  task_cls = _TASKS[task_name]
  return task_cls(**config)


def run(config):
  config = OmegaConf.to_container(config, resolve=True)
  logfile = os.path.join("results.jsonl")
  logger = JsonLogger(filename=logfile)
  task = _build_task(config["task"])
  method_config = config["method"]
  method_name = method_config.pop("name")
  if method_name == "dmc":
    methods.run_dmc(
        task,
        seed=config['seed'],
        num_samples=config["num_samples"],
        **method_config,
        logger=logger)
  elif method_name == "stratified":
    methods.run_stratified(
        task,
        seed=config['seed'],
        num_samples=config["num_samples"],
        **method_config,
        logger=logger)
  elif method_name == "pimais":
    methods.run_sympais(
        task,
        key=jax.random.PRNGKey(config['seed']),
        num_samples=config["num_samples"],
        **method_config,
        logger=logger)
  elif method_name == "hpimais":
    methods.run_sympais_hmc(
        task,
        key=jax.random.PRNGKey(config['seed']),
        num_samples=config["num_samples"],
        **method_config,
        logger=logger)
  else:
    raise ValueError(f"Unknown method: {method_name}")
  logger.close()


@hydra.main(config_path='conf', config_name="config")
def main(cfg):
  # import multiprocessing as mp
  # Limit the number of threads used
  os.environ["OMP_NUM_THREADS"] = "2"
  os.environ["OPENBLAS_NUM_THREADS"] = "2"
  os.environ["MKL_NUM_THREADS"] = "2"
  os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
  os.environ["NUMEXPR_NUM_THREADS"] = "2"
  # Limit the no. of threads used by JAX
  os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                             "intra_op_parallelism_threads=2")
  log.info("Config:\n %s", OmegaConf.to_yaml(cfg))
  log.info("Working directory : {}".format(os.getcwd()))
  if cfg["disable_jit"]:
    import jax
    with jax.disable_jit():
      run(cfg)
  run(cfg)


if __name__ == "__main__":
  main()  # pylint: disable=all
