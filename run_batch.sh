#!/bin/sh

#SBATCH --time=1:00:00
#SBATCH --ntasks=
#SBATCH --mem=50gb
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=12355
echo "Running Parallelized Batch Training"


#Paste in the line you want to run below
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 torch_train_DPP.py --loss categorical --model transformer --nepochs 10 --ipath /work/tier3/jkrupa/zprlegacy/process/19May23_physicalpt_v2_with2018_v2/merged/train/raw/ --vpath /work/tier3/jkrupa/zprlegacy/process/19May23_physicalpt_v2_with2018_v2/merged/val/raw/ --opath dummy_test_v0 --nparts 100 --nclasses 4 --mini_dataset --batchsize 500 --embedding_size 32 --hidden_size 32 --num_attention_heads 4 --intermediate_size 32 --num_hidden_layers 2 --feature_size 13 --nclasses 4 --num_encoders 6 --n_out_nodes 20 --plot_text "Transformer; dummy" --num_max_files 10 --sv --feature_sv_size 16

