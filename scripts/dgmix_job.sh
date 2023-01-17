#!/bin/bash

stage=1
start_stage=3
available_gpus=12

tag=dgmix.original
seed=0
machines="fan gearbox bearing valve slider ToyCar ToyTrain"
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
# shellcheck disable=SC1091
. utils/original_funcs.sh || exit 1
set -euo pipefail
epochs="50 100 120"

if [ "${stage}" -le 0 ] && [ "${stage}" -ge 0 ]; then
    log "Prepare datasets."
    log "This command should be executed only the first time."
    sbatch -p ubuntu ./dgmix_run.sh \
        --stage "1" \
        --stop_stage "2" \
        --seed "${seed}"
fi

if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    for machine in ${machines}; do
        slurm_gpu_scheduler "${available_gpus}"
        log "Start model training ${machine}/${tag}_seed${seed}."
        sbatch -p ubuntu ./dgmix_run.sh \
            --stage "${start_stage}" \
            --stop_stage "5" \
            --pos_machine "${machine}" \
            --tag "${tag}" \
            --epochs "${epochs}" \
            --seed "${seed}"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    ./local/scoring.sh \
        --no "${tag}/seed${seed}" \
        --epochs "${epochs}"
fi
