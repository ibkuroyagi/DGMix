#!/bin/bash

# Copyright 2023 Ibuki Kuroyanagi
# shellcheck disable=SC1091
. ./cmd.sh || exit 1
. ./path.sh || exit 1

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# basic settings
stage=3      # stage to start
stop_stage=5 # stage to stop
verbose=1    # verbosity level (lower is less info)
seed=0       # Seed for all setting
# data split setting
valid_percent=15 # Ratio of validation data

# training related

pos_machine=fan
# directory path setting
expdir=exp
# training related setting
tag=dgmix.original # tag for directory to save model
# inference related setting
epochs="50 100 150"
checkpoints=""
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
set -euo pipefail
conf="conf/tuning/${tag}.yaml"
log "Start run.sh"

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Download data."
    local/download_data.sh downloads
    log "Successfully downloaded data."
    # shellcheck disable=SC2154
    ${train_cmd} "downloads/eval/rename.log" \
        python local/rename.py
    log "Successfully rename data."
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Distribute data."
    # shellcheck disable=SC2154
    ${train_cmd} "downloads/dev/distribute_data_seed${seed}.log" \
        python local/distribute_data.py \
        --seed "${seed}" \
        --valid_percent "${valid_percent}"
    log "Successfully distrubuted data."
fi

attribute_csv="downloads/dev/${pos_machine}/attributes_seed${seed}.csv"
outdir="${expdir}/${pos_machine}/${tag}/seed${seed}"
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Stage 3: Train a DGMix model's feature extractor."
    log "Training start. See the progress via ${outdir}/train_${pos_machine}_${tag}_seed${seed}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} --gpu 1 "${outdir}/train_${pos_machine}_${tag}_seed${seed}.log" \
        python -m asd_tools.bin.train \
        --attribute_csv "${attribute_csv}" \
        --outdir "${outdir}" \
        --config "${conf}" \
        --seed "${seed}" \
        --verbose "${verbose}"
    log "Successfully finished training."
fi
# shellcheck disable=SC2086
if [ -z ${checkpoints} ]; then
    checkpoints+="${outdir}/best_loss/best_loss.pkl "
    for epoch in ${epochs}; do
        checkpoints+="${outdir}/checkpoint-${epoch}epochs/checkpoint-${epoch}epochs.pkl "
    done
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Stage 4: Embedding calculation start. See the progress via ${outdir}/embed_${pos_machine}_${tag}_seed${seed}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} --gpu 1 "${outdir}/embed_${pos_machine}_${tag}_seed${seed}.log" \
        python -m asd_tools.bin.embed \
        --attribute_csv "${attribute_csv}" \
        --checkpoints ${checkpoints} \
        --config "${conf}" \
        --verbose "${verbose}"
    log "Successfully finished extracting embedding."
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    log "Stage 5: Inference start. See the progress via ${outdir}/infer_${pos_machine}_${tag}_seed${seed}.log."
    # shellcheck disable=SC2154,SC2086
    ${cuda_cmd} "${outdir}/infer_${pos_machine}_${tag}_seed${seed}.log" \
        python -m asd_tools.bin.infer \
        --checkpoints ${checkpoints} \
        --config "${conf}" \
        --attribute_csv "${attribute_csv}" \
        --seed "${seed}" \
        --verbose "${verbose}"
    log "Successfully finished Inference."
fi
