#!/bin/bash

stage=1
start_stage=3

tag=dgmix.original
seed=0
machines="fan gearbox bearing valve slider ToyCar ToyTrain"
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y/%m/%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

set -euo pipefail
epochs="50 100 120"

if [ "${stage}" -le 1 ] && [ "${stage}" -ge 1 ]; then
    log "Prepare datasets."
    log "This command should be executed only the first time."
    ./run.sh \
        --stage "1" \
        --stop_stage "2" \
        --seed "${seed}"
fi

if [ "${stage}" -le 2 ] && [ "${stage}" -ge 2 ]; then
    for machine in ${machines}; do
        log "Start model training ${machine}/${tag}_seed${seed}."
        ./run.sh \
            --stage "${start_stage}" \
            --stop_stage "5" \
            --pos_machine "${machine}" \
            --tag "${tag}" \
            --epochs "${epochs}" \
            --seed "${seed}"
    done
fi

if [ "${stage}" -le 3 ] && [ "${stage}" -ge 3 ]; then
    ./local/scoring.sh \
        --no "${tag}/seed${seed}" \
        --epochs "${epochs}"
fi
