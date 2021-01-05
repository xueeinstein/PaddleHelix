#!/bin/bash
cd $(dirname $0)

batch_size="256"
lr="0.001"
max_epoch="100"
thread_num="8" # thread_num is for cpu, please set CUDA_VISIBLE_DEVICES for gpu
warmup_steps="0"
init_model="/mnt/xueyang/Code/PaHelix/apps/mat/demos/model_dir/pretrain_config_ffn1/best_model"
use_cuda="true" # candidates: true/false

model_config="$1"
data_npz="$2"
data_split="$3"
model_dir="$4"
cuda_device="$5"  # only work when ${use_cuda}="true"

if [ "${use_cuda}" == "true" ]; then
    export CUDA_VISIBLE_DEVICES=${cuda_device}
    python ../train.py \
            --use_cuda \
            --batch_size ${batch_size} \
            --warmup_steps ${warmup_steps} \
            --lr ${lr} \
            --max_epoch ${max_epoch} \
            --data_npz ${data_npz} \
            --data_split ${data_split} \
            --model_config ${model_config} \
            --init_model ${init_model} \
            --model_dir ${model_dir}
else
    python ../train.py \
            --batch_size ${batch_size} \
            --warmup_steps ${warmup_steps} \
            --lr ${lr} \
            --max_epoch ${max_epoch} \
            --thread_num ${thread_num} \
            --data_dir ${data_dir} \
            --data_split ${data_split} \
            --model_config ${model_config} \
            --init_model ${init_model} \
            --model_dir ${model_dir}
fi
