#!/bin/bash

# This script is used for Stage 2 task adaptation of MixDA.
# You can specify the following parameters.

devices=$1 # number of GPUs used
shots=$2 # for k-shot learning
dataset=$3 
lr=$4 # learning rate
max_epochs=$5
batch_size=$6 # batch size
type=$7 # task adapter type. for example, houlsby (w/o domain adapter), houlsby-mixda (with domain adapter included), finetune, etc.
stage_one_path=$8 # path to stage 1 domain adapters
train_path=$9
dev_path=$10
test_path=$11
project_name=$12 # project name for wandb



# $1: dataset, $2: lr, $3: batch, $4: type, $5: seed
run_others() {
    run_name=${1}_${4}_${3}_${2}_seed${5}
    if [[ ! -d ./results/${run_name} ]]; then
        mkdir -p ./results/${run_name}
    fi

    add_arguments=""
    if [[ $4 == *"-mixda" ]]; then
        add_arguments="${add_arguments} --load_adapters ${stage_one_path} --layers 7,11"
    fi
    if [[ ${4%-*} != "finetune" ]]; then
        add_arguments="${add_arguments} --adapter_type ${4%-*} --reduction_factor 16 --adapter_non_linearity swish"
    fi
    if [[ $shots != 0 ]]; then
        add_arguments="${add_arguments} --few_shot $shots"
    fi

   WANDB_PROJECT=${project_name} WANDB_NAME=${run_name}  \
    python3 -m scripts.run_others --max_epochs=${max_epochs} --accelerator gpu --devices $devices --batch_size $3 --project_name ${project_name} --run_name $run_name --lr ${lr} --train_data_path $train_path --dev_data_path $dev_path \
    --test_data_path $test_path --dirpath ./results/${run_name} $add_arguments
}

# $1: dataset, $2: lr, $3: batch, $4: type, $5: seed
run_glue() {
    # send parameters to Huggingface accelerate
    printf "0\n2\n1\nno\nno\n${devices}\n\nno\n" | accelerate config
    run_name=${1}_${4}_${3}_${2}_seed${5}

    add_arguments=""
    if [[ $4 == *"-mixda" ]]; then
        add_arguments="${add_arguments} --load_mixda ${stage_one_path} --layers 7,11"
    fi
    if [[ ${4%-*} != "finetune" ]]; then
        add_arguments="${add_arguments} --adapter_type ${4%-*} --reduction_factor 16 --adapter_non_linearity swish"
    fi
    if [[ $shots != 0 ]]; then
        add_arguments="${add_arguments} --few_shot $shots"
    fi

    WANDB_PROJECT=${project_name} WANDB_NAME=${run_name}  \
        accelerate launch scripts/run_glue.py --task_name $1 --model_name_or_path roberta-large \
        --num_train_epochs ${max_epochs} --report_to wandb \
        --with_tracking --learning_rate $2 --checkpointing_steps=epoch --output_dir=results/${run_name} \
        --project_name ${project_name} --per_device_train_batch_size ${3} $add_arguments 
}

# $1: dataset, $2: lr, $3: batch, $4: type, $5: seed
run_csqa() {
    get_ka_type $1
    run_name=${1}_${4}_${3}_${2}_seed${5}
    if [[ ! -d ./results/${run_name} ]]; then
        mkdir -p ./results/${run_name}
    fi

    add_arguments=""
    if [[ $4 == *"-mixda" ]]; then
        add_arguments="${add_arguments} --load_mixda ${stage_one_path} --layers 7,11"
    fi
    if [[ ${4%-*} != "finetune" ]]; then
        add_arguments="${add_arguments} --adapter_type ${4%-*} --reduction_factor 16 --adapter_non_linearity swish"
    fi
    if [[ $shots != 0 ]]; then
        add_arguments="${add_arguments} --few_shot $shots"
    fi

    WANDB_PROJECT=${project_name} \
    python3 scripts/run_csqa.py --max_epochs=${max_epochs} --accelerator gpu --strategy ddp --devices $devices --batch_size $3 --dirpath results/${run_name} --project_name ${project_name} --run_name $run_name --lr $2 $add_arguments 
}

# $1: dataset, $2: lr, $3: batch, $4: type, $5: seed
run_one() {
    dataset=$1
    case "$dataset" in
        csqa)
            run_csqa $1 $2 $3 $4 $5
            ;;
        chemprots|sciie|citation_intent|rct|hyperpartisan|imdb|ag|amazon|fever)
            run_others $1 $2 $3 $4 $5
            ;;
        *)
            run_glue $1 $2 $3 $4 $5
            ;;
    esac
}


seed=$RANDOM
run_one $dataset $lr $batch_size $type $seed
