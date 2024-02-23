#!/usr/bin/env bash
# -*- coding:utf-8 -*-
hostname
python -u -c 'import torch; print(torch.__version__); print(torch.cuda.device_count())'
nvidia-smi

export exp_name="output-llama-7b-json"
export batch_size=4
export gradient_accumulation_steps=16
export model_name=yahma/llama-7b-hf
export model_type=llama
export data_name="flan,ie"
export lr=1e-3
export seed=42
export CUDA_VISIBLE_DEVICES=0
export lr_scheduler=linear
export epochs=3
export verbose=False
export fp16=''
export max_length=2048
export max_num_instances_flan=50000
export max_num_instances_ie=5000
export add_output_control=1
export add_label_space=1
export text_tuning=0
export lora_target_modules="q_proj,v_proj"
export lora_r=8
export lora_alpha=16
export lora_dropout=0.05

OPTS=$(getopt -o b:d:m:i:t:k:s:l:f:n:v --long batch:,device:,text_tuning:,model:,model_type:,data:,seed:,lr:,lora_target_modules:,lora_r:,lora_alpha:,lora_dropout:,epochs:,gradient_accumulation_steps:,exp_name:,verbose,max_num_instances_flan:,max_num_instances_ie:,max_length:, -n 'parse-options' -- "$@")

if [ $? != 0 ]; then
  echo "Failed parsing options." >&2
  exit 1
fi

eval set -- "$OPTS"

while true; do
  case "$1" in
  -b | --batch)
    batch_size="$2"
    shift
    shift
    ;;
  -d | --device)
    CUDA_VISIBLE_DEVICES="$2"
    shift
    shift
    ;;
  -m | --model)
    model_name="$2"
    shift
    shift
    ;;
  -i | --data)
    data_name="$2"
    shift
    shift
    ;;
  -s | --seed)
    seed="$2"
    shift
    shift
    ;;
  -l | --lr)
    lr="$2"
    shift
    shift
    ;;
  --model_type)
    model_type="$2"
    shift
    shift
    ;;
  --text_tuning)
    text_tuning="$2"
    shift
    shift
    ;;
  --lora_target_modules)
    lora_target_modules="$2"
    shift
    shift
    ;;
  --lora_r)
    lora_r="$2"
    shift
    shift
    ;;
  --lora_alpha)
    lora_alpha="$2"
    shift
    shift
    ;;
  --lora_dropout)
    lora_dropout="$2"
    shift
    shift
    ;;
  --epochs)
    epochs="$2"
    shift
    shift
    ;;
  --gradient_accumulation_steps)
    gradient_accumulation_steps="$2"
    shift
    shift
    ;;
  --exp_name)
    exp_name="$2"
    shift
    shift
    ;;
  --max_num_instances_flan)
    max_num_instances_flan="$2"
    shift
    shift
    ;;
  --max_num_instances_ie)
    max_num_instances_ie="$2"
    shift
    shift
    ;;
  --max_length)
    max_length="$2"
    shift
    shift
    ;;
  -v | --verbose)
    verbose=True
    shift
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "$1" not recognize.
    exit
    ;;
  esac
done


get_gpu_num() {
  IFS=,
  num=0
  for i in ${CUDA_VISIBLE_DEVICES}
  do
    num=$((${num} + 1))
  done
  echo ${num}
  return ${num}
}

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))
}

gpu_num=$(get_gpu_num)
# For multiple GPU, use the Distributed version of PyTorch
if [[ ${gpu_num} == 1 ]]
then
    run_command=python3
else
    master_port=$(rand 10000 50000)
    echo "Master Port: ${master_port}"
    run_command="python3 -m torch.distributed.launch --nproc_per_node ${gpu_num} --master_port ${master_port}"
fi

echo "run_command: " ${run_command}
echo "exp_name: " ${exp_name}

batch_log=$((gpu_num * batch_size * gradient_accumulation_steps))

if [[ ${text_tuning} == 1 ]]
then
    output_dir=${exp_name}/${data_name}_lr${lr}_e${epochs}_ml${max_length}_mf${max_num_instances_flan}_mi${max_num_instances_ie}_b${batch_log}_lora-${lora_target_modules}-${lora_r}-${lora_alpha}-${lora_dropout}
else
    output_dir=${exp_name}/${data_name}_lr${lr}_e${epochs}_ml${max_length}_mf${max_num_instances_flan}_mi${max_num_instances_ie}_b${batch_log}_lora-${lora_target_modules}-${lora_r}-${lora_alpha}-${lora_dropout}_aoc${add_output_control}_als${add_label_space}
fi

if [[ ! -d ${exp_name} ]]
then
  mkdir ${exp_name}
fi

if [[ ! -d ${output_dir} ]]
then
  mkdir ${output_dir}
fi

export PYTHONPATH="${PYTHONPATH}:./"
