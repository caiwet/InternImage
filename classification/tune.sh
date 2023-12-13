#!/bin/bash
#SBATCH --job-name=cls_train_ranzcr             # Specify your job name here
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-40:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_requeue                     # Partition to run in
#SBATCH --gres=gpu:1
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)
#SBATCH --account=rajpurkar_prr712
#SBATCH -o cls_train_ranzcr%j.out                          # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e cls_train_ranzcr%j.err                          # File to which STDERR will be written, including job ID (%j)

module load miniconda3/4.10.3 gcc/9.2.0 cuda/11.7
source activate internimage
cd /home/cat302/ETT-Project/InternImage/classification
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/internimage_t_1k_224.yaml --data-path /n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/all_data_split --batch-size 4 --output $SLURM_JOB_NAME
