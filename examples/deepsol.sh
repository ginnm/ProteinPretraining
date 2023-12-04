#PBS -q ai
#PBS -l select=1:ncpus=12:ngpus=1:mem=64gb
#PBS -l walltime=168:00:00
#PBS -m abe

source /home/limc/miniconda3/bin/activate peta
cd /home/limc/workspace/ProteinPretraining

export HTTP_PROXY="http://g5:15777"
export HTTPS_PROXY="http://g5:15777"
export PYTHONPATH="$PYTHONPATH:./"

DATASET="deepsol"
BATCH_SIZE=32
MODEL="ElnaggarLab/ankh-base"
POOLING_HEAD="attention1d"
DEVICES=1
NUM_NODES=1
SEED=3407
PRECISION='bf16'
MAX_EPOCHS=20
ACC_BATCH=1
LR=0.0001
PATIENCE=4
STRATEGY="auto"
FINETUNE="head"

python peta/train.py \
--dataset $DATASET \
--batch_size $BATCH_SIZE \
--model $MODEL \
--pooling_head $POOLING_HEAD \
--devices $DEVICES \
--strategy $STRATEGY \
--num_nodes $NUM_NODES \
--seed $SEED \
--precision $PRECISION \
--max_epochs $MAX_EPOCHS \
--accumulate_grad_batches $ACC_BATCH \
--lr $LR \
--patience $PATIENCE \
--finetune $FINETUNE \
--wandb_project enzyme-final-$DATASET \
--wandb