#!/bin/bash

#SBATCH --partition gpu
#SBATCH -c 4
#SBATCH --mail-type=FAIL,RUNNING,COMPLETE
#SBATCH --mail-user=ctian@fas.harvard.edu
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH --time=80:00:00
#SBATCH -o ../bash_output/sbatch_combine_%j_run.out
#SBATCH -e ../bash_output/sbatch_combine_%j_err.out

source ~/.bashrc

module load miniconda3/4.10.3
module load gcc/6.2.0
module load cuda/11.2
source activate internimage

cd /home/cat302/ETT-Project/InternImage/detection

srun -c 1 hostname

srun -c 1 echo "Hello, I ran an sbatch script with srun commands!"