#!/usr/bin/env bash
set -xv

common_args="\
    hydra.launcher.n_jobs=10 \
    seed=`seq -s , 0 19` \
    task=torus \
    hydra.sweep.subdir=num_samples-\${num_samples}_profile-\${task.profile_type}_seed-\${seed} \
    hydra.sweep.dir='multirun/esec-fse/\${task.name}/\${method.name}'\
    "

# Independent
# DMC groundtruth
python src/sympais/experiments/run.py -m \
    "method=dmc" \
    'num_samples=1_000_000_00' \
    "method.batch_size=5_000_000" \
    $common_args

# DMC baseline
python src/sympais/experiments/run.py -m \
    "method=dmc" \
    'num_samples=1_000_000' \
    "method.batch_size=50_000" \
    $common_args

# qCoral
python src/sympais/experiments/run.py -m \
    "method=stratified" \
    'num_samples=1_000_000' \
    "method.batch_size=50_000" \
    $common_args

# PIMAIS
python src/sympais/experiments/run.py -m \
    "method=pimais" \
    'num_samples=1_000_000' \
    task=torus \
    method.tune=false \
    method.init=realpaver \
    method.num_warmup_steps=500 \
    method.proposal_scale_multiplier=0.5 \
    $common_args

# HPIMAIS
python src/sympais/experiments/run.py -m \
    "method=hpimais" \
    'num_samples=1_000_000' \
    task=torus \
    method.init=realpaver \
    method.num_warmup_steps=500 \
    method.proposal_scale_multiplier=0.5 \
    $common_args

# Correlated
# DMC groundtruth
python src/sympais/experiments/run.py -m \
    "method=dmc" \
    task.profile_type=correlated \
    task.scale=.25 \
    'num_samples=1_000_000_00' \
    "method.batch_size=5_000_000" \
    $common_args

# DMC baseline
python src/sympais/experiments/run.py -m \
    "method=dmc" \
    task.profile_type=correlated \
    task.scale=.25 \
    'num_samples=1_000_000' \
    "method.batch_size=50_000" \
    $common_args

python src/sympais/experiments/run.py -m \
    task=torus \
    task.profile_type=correlated \
    task.scale=.25 \
    "method=pimais" \
    'num_samples=1_000_000' \
    method.tune=false \
    method.init=realpaver \
    method.resample=false \
    method.num_warmup_steps=500 \
    method.proposal_scale_multiplier=0.5 \
    $common_args

python src/sympais/experiments/run.py -m \
    task=torus \
    task.profile_type=correlated \
    task.scale=.25 \
    "method=hpimais" \
    'num_samples=1_000_000' \
    method.init=realpaver \
    method.resample=false \
    method.num_warmup_steps=500 \
    method.proposal_scale_multiplier=0.5 \
    $common_args
