#!/usr/bin/env bash
# -*- coding:utf-8 -*-

device=$1
checkpoint_path=$2
batch_size=$3
use_all_templates=$4


get_gpu_num() {
  IFS=,
  num=0
  for i in ${device}
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

#checkpoint_path=output-llama-7b-json/flan,ie_lr1e-3_e3_ml2048_mf50000_mi5000_b64_lora-q_proj,v_proj-8-16-0.05_aoc1_als1

CUDA_VISIBLE_DEVICES=${device} \
  ${run_command} JsonEvaluation_mmlu.py \
      --model_type llama \
      --data mmlu \
      --per_gpu_eval_batch_size ${batch_size} \
      --max_num_instances_mmlu -1 \
      --use_all_templates ${use_all_templates} \
      --resume_from_checkpoint ${checkpoint_path} \

CUDA_VISIBLE_DEVICES=${device} \
  ${run_command} JsonEvaluation_bbh.py \
      --model_type llama \
      --data bbh \
      --per_gpu_eval_batch_size ${batch_size} \
      --max_num_instances_bbh -1 \
      --use_all_templates ${use_all_templates} \
      --resume_from_checkpoint ${checkpoint_path} \

CUDA_VISIBLE_DEVICES=${device} \
  ${run_command} JsonEvaluation_ner.py \
      --model_type llama \
      --data ner \
      --per_gpu_eval_batch_size ${batch_size} \
      --max_num_instances_ner 500 \
      --use_all_templates ${use_all_templates} \
      --resume_from_checkpoint ${checkpoint_path} \

CUDA_VISIBLE_DEVICES=${device} \
${run_command} JsonEvaluation_re.py \
    --model_type llama \
    --data re \
    --per_gpu_eval_batch_size ${batch_size} \
    --max_num_instances_re 500 \
    --use_all_templates ${use_all_templates} \
    --resume_from_checkpoint ${checkpoint_path} \

CUDA_VISIBLE_DEVICES=${device} \
${run_command} JsonEvaluation_ee.py \
    --model_type llama \
    --data ee \
    --per_gpu_eval_batch_size ${batch_size} \
    --max_num_instances_ee 500 \
    --use_all_templates ${use_all_templates} \
    --resume_from_checkpoint ${checkpoint_path} \

CUDA_VISIBLE_DEVICES=${device} \
  ${run_command} JsonEvaluation_sql.py \
      --model_type llama \
      --data sql \
      --per_gpu_eval_batch_size ${batch_size} \
      --max_num_instances_sql 500 \
      --use_all_templates ${use_all_templates} \
      --resume_from_checkpoint ${checkpoint_path} \

