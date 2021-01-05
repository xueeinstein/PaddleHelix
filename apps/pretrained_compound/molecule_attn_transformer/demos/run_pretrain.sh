#!/bin/bash
cd $(dirname $0)

# model_config="pretrain_config.json"
model_config="$1"

batch_size="32"
lr="0.001"
max_epoch="30"
thread_num="8" # thread_num is for cpu, please set CUDA_VISIBLE_DEVICES for gpu
warmup_steps="0"
model_dir="model_dir/pretrain_on_zinc"
use_cuda="true" # candidates: true/false
cuda_devices="0,1,2,3,4,5,6,7"  # only work when ${use_cuda}="true"
distributed="true" # candidates: true/false
data_dir="/mnt/xueyang/Datasets/PaddleHelix/mat/zinc15/"
data_npz="/mnt/xueyang/Datasets/PaddleHelix/mat/zinc.npz"
# data_npz="small.npz"

if [ "${distributed}" == "true" ]; then
    if [ "${use_cuda}" == "true" ]; then
        export FLAGS_sync_nccl_allreduce=1
        export FLAGS_fuse_parameter_memory_size=64
        export CUDA_VISIBLE_DEVICES=${cuda_devices}

        python -m paddle.distributed.launch \
               --log_dir log_dir/pretrain_on_zinc \
               ../train.py \
               --use_cuda \
               --batch_size ${batch_size} \
               --warmup_steps ${warmup_steps} \
               --lr ${lr} \
               --max_epoch ${max_epoch} \
               --data_dir ${data_dir} \
               --model_config ${model_config} \
               --model_dir ${model_dir} \
               --pretrain \
               --distributed
    else
        echo "Only gpu is supported for distributed mode at present."
    fi
else
    if [ "${use_cuda}" == "true" ]; then
        export CUDA_VISIBLE_DEVICES="2"
        python ../train.py \
               --use_cuda \
               --batch_size ${batch_size} \
               --warmup_steps ${warmup_steps} \
               --lr ${lr} \
               --max_epoch ${max_epoch} \
               --data_dir ${data_dir} \
               --model_config ${model_config} \
               --model_dir ${model_dir} \
               --pretrain
    else
        python ../train.py \
               --batch_size ${batch_size} \
               --warmup_steps ${warmup_steps} \
               --lr ${lr} \
               --max_epoch ${max_epoch} \
               --thread_num ${thread_num} \
               --data_dir ${data_dir} \
               --model_config ${model_config} \
               --model_dir ${model_dir} \
               --pretrain
    fi
fi
