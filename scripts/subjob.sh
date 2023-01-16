#!/bin/bash

./disentangle_job.sh --machines "ToyTrain" --seed 3 --n_domain -1 --start_stage 3
./disentangle_job.sh --machines "gearbox" --seed 2 --n_domain -1 --start_stage 4
./disentangle_job.sh --machines "ToyTrain" --seed 2 --n_domain 0 --start_stage 3
./disentangle_job.sh --machines "valve" --seed 0 --n_domain 15 --start_stage 3
./disentangle_job.sh --machines "ToyCar" --seed 3 --n_domain 20 --start_stage 3
