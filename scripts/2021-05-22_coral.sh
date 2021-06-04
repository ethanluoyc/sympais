#!/usr/bin/env bash
set -xv

common_args="\
    hydra.launcher.n_jobs=20 \
    seed=`seq -s , 0 4` \
    hydra.sweep.subdir=num_samples-\${num_samples}_subject-\${task.filename}_path-\${task.path_index}_seed-\${seed} \
    hydra.sweep.dir='multirun/esec-fse/\${task.name}/\${method.name}' \
    task=coral \
    task.filename=example-cart-12.m,example-carton-5-0.m,example-ckd-epi-0.m,example-ckd-epi-simple-0.m,framingham-0.m \
    task.path_index=0,1,2,3,4"

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
    method.init=realpaver \
    method.num_warmup_steps=500 \
    method.proposal_scale_multiplier=0.5 \
    disable_jit=true \
    $common_args

# HPIMAIS
# src/sympais/experiments/run.py -m \
#     "method=hpimais" \
#     'num_samples=1_000_000' \
#     method.tune=False \
#     method.init=z3,realpaver \
#     method.num_warmup_steps=500 \
#     method.proposal_scale_multiplier=0.5 \
#     $common_args
