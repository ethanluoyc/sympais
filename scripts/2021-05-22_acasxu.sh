#!/usr/bin/env bash
set -xv

common_args="\
    hydra.launcher.n_jobs=10 \
    seed=`seq -s , 0 19` \
    task=acasxu \
    task.path_index=`seq -s , 0 5` \
    hydra.sweep.subdir=num_samples-\${num_samples}_path-\${task.path_index}_seed-\${seed} \
    hydra.sweep.dir='multirun/esec-fse-acasxu/\${task.name}/\${method.name}'\
    "

# DMC groundtruth
python src/sympais/experiments/run.py -m \
    "method=dmc" \
    'num_samples=10_000_000' \
    "method.batch_size=1_000_000" \
    $common_args

# DMC baseline
python src/sympais/experiments/run.py -m \
    "method=dmc" \
    'num_samples=1_000_000' \
    "method.batch_size=5_000_0" \
    $common_args

# qCoral
python src/sympais/experiments/run.py -m \
    "method=stratified" \
    'num_samples=1_000_000' \
    "method.batch_size=5_000_0" \
    $common_args

# PIMAIS
python src/sympais/experiments/run.py -m \
    "method=pimais" \
    'num_samples=1_000_000' \
    method.tune=true \
    method.init=z3 \
    method.num_warmup_steps=500 \
    method.proposal_scale_multiplier=0.5 \
    $common_args

# HPIMAIS
python src/sympais/experiments/run.py -m \
    "method=hpimais" \
    'num_samples=1_000_000' \
    method.init=z3 \
    method.num_warmup_steps=500 \
    method.proposal_scale_multiplier=0.5 \
    $common_args
