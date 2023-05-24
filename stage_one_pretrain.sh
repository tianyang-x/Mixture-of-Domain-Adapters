#!/bin/bash

# This script is used for Stage 1 pretraining of MixDA.
# You can specify the following parameters.

dirname=$1 # the directory to save the pre-trained model
knowledge_path=$2 # path to domain knowledge
project_name=$3 # the name of the project
max_epochs=$4 # the maximum number of epochs
devices=$5 # number of GPUs used
batch_size=$6 # batch size
lr=$7 # learning rate
moa_lr=$8 # learning rate for MoA
realm_record=$9 # path to the REALM record used for information retrieval. Ignore it to disable it
no_old_knowledge=$10 # set it to 1 if you want to disale old domain knowledge

other_params=""
if [[ $no_old_knowledge -eq 1 ]]; then
    other_params="--no_old_knowledge"
fi
if [[ $realm_record != "" ]]; then
    other_params="${other_params} --realm_record ${realm_record}"
fi

python -m scripts.run_stage_one \
    --max_epochs ${max_epochs} \
    --accelerator gpu --strategy ddp \
    --devices ${devices} \
    --batch_size ${batch_size} \
    --layers 7,11 \
    --knowledge_data_path ${knowledge_path} \
    --project_name ${project_name} \
    --run_name ${project_name} \
    --dirpath ${dirname} \
    --lr ${lr} \
    --moe_lr ${moa_lr} \
    --adapter_down_scale 16
    --realm_record ${realm_record} \
    ${other_params}

