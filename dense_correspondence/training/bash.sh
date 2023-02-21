#!/bin/bash
# +
#!/bin/bash
#SBATCH -o shirt_vert-pose_white_sizes_%j.log
#SBATCH -c 4
#SBATCH --gres=gpu:volta:1
# -

# Initialize module command
source /etc/profile

# Loading modules
module load anaconda/2022a
module load cuda/11.0

# Run the script
echo "Starting: $SLURM_ARRAY_TASK_ID"
python3 batch_training.py
