#!/bin/bash
#SBATCH --job-name=your-job-name
#SBATCH --partition=gpu
#SBATCH --time=72:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=p40&gmem24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --chdir=/scratch/shared/beegfs/your_dir/
#SBATCH --output=/scratch/shared/beegfs/your_dir/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=8

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv

### the command to run
srun python3 torch_train_DDP_v2.py --loss categorical --model transformer --nepochs 10 --ipath /work/tier3/jkrupa/zprlegacy/process/19May23_physicalpt_v2_with2018_v2/merged/train/raw/ --vpath /work/tier3/jkrupa/zprlegacy/process/19May23_physicalpt_v2_with2018_v2/merged/val/raw/ --opath dummy_test_v0 --nparts 100 --nclasses 4 --mini_dataset --batchsize 500 --embedding_size 32 --hidden_size 32 --num_attention_heads 4 --intermediate_size 32 --num_hidden_layers 2 --feature_size 13 --nclasses 4 --num_encoders 6 --n_out_nodes 20 --plot_text "Transformer; dummy" --num_max_files 10 --sv --feature_sv_size 16
