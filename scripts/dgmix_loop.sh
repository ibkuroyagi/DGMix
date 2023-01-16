#!/bin/bash

for n_domain in -1 0 5 10 15 20 25; do
    for seed in 0 1 2 3 4; do
        ./dgmix_job.sh --n_domain "${n_domain}" --seed "${seed}" --tag dgmix.original --start_stage 3
    done
done

for n_domain in -1 0 5 10 15 20 25; do
    for seed in 0 1 2 3 4; do
        sbatch ./dgmix_job.sh --n_domain "${n_domain}" --seed "${seed}" --tag dgmix.original --stage 2
    done
done
