#!/bin/bash
#SBATCH --job-name=pretrained-no-ett-3           # Specify your job name here
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-20:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_requeue                     # Partition to run in
#SBATCH --gres=gpu:4
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)
#SBATCH --account=rajpurkar_prr712
#SBATCH -o ranzcr-pretrained-mimic02%j.out                          # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ranzcr-pretrained-mimic02%j.err                          # File to which STDERR will be written, including job ID (%j)

module load miniconda3/4.10.3 gcc/9.2.0 cuda/11.7
source activate internimage
cd /n/scratch3/users/c/cat302/ETT-Project/InternImage/detection
# python train.py configs/ett/cascade_rcnn_internimage_t_fpn_1x_ett.py --work-dir ./work_dirs_$SLURM_JOB_NAME
sh dist_train.sh configs/ett/cascade_rcnn_internimage_t_fpn_1x_ett.py 4 ./work_dirs_$SLURM_JOB_NAME
