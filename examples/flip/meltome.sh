export HTTP_PROXY="http://g5:15777"
export HTTPS_PROXY="http://g5:15777"
export PYTHONPATH="$PYTHONPATH:./"

DATASET="meltome"
POOLING_HEAD="attention1d"
DEVICES=1
NUM_NODES=1
SEED=3407
PRECISION='bf16'
STRATEGY="auto"
FINETUNE="head"

python peta/train.py \
--dataset $DATASET \
--batch_size $BATCH_SIZE \
--model $MODEL \
--model_type $MODEL_TYPE \
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