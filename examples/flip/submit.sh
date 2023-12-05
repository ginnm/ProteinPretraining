# aav
splits="two_vs_many seven_vs_many sampled raw one_vs_many mut_des low_vs_high des_mut"


# gb1
splits="low_vs_high one_vs_rest raw sampled three_vs_rest two_vs_rest"


# meltome
splits="human human_cell mixed_split raw"


function get_model_type {
    local x="$1"
    if [ "$x" == "AI4Protein/meer_base" ]; then
        echo "debert"
    elif [ "$x" == "AI4Protein/esm2_650M_MEER" ]; then
        echo "esm"
    else
        echo "Unknown"  # 或者返回空字符串或特定错误信息
    fi
}

# meltome
for model_path in AI4Protein/meer_base
do
    for LR in 0.001 0.0001 0.00001
    do
        for BATCH_SIZE in 16 32
        do
            model=$model_path
            model_type=$(get_model_type $model_path)
            MAX_EPOCHS=200
            ACC_BATCH=4
            LR=0.0001
            BATCH_SIZE=32
            PATIENCE=20
            echo "sbatch -J meltome-$model_type \
            -v MODEL=$model,MODEL_TYPE=$model_type,MAX_EPOCHS=$MAX_EPOCHS,ACC_BATCH=$ACC_BATCH,LR=$LR,BATCH_SIZE=$BATCH_SIZE,PATIENCE=$PATIENCE \
            -o logs/meltome-$model_type-%j.out \
            -e logs/meltome-$model_type-%j.err \
            examples/flip/meltome.sh"
        done
    done
done
