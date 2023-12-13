# Deep Intubation

This is the repo for Deep Intubation project - endotracheal tube detection on chest X-rays.

## Get Started

We use InternImage + gloria. 

1. `detection/bash/tune.sh` contains the command to train submit a job on O2. The default is to use 4 GPUs. Change `--job-name` to the name you want. 

2. Config model under `detection/configs`.