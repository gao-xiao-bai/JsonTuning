#!/usr/bin/env bash
# -*- coding:utf-8 -*-

source function_code.bash
#export verbose=True

if [[ ${verbose} == True ]]
then
  stdout_file=/dev/stdout
  stderr_file=/dev/stderr
else
  stdout_file=${output_dir}/run.log
  stderr_file=${output_dir}/run.err
fi

#CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
${run_command} JsonTuning.py \
    --seed ${seed} \
    --model_type ${model_type} \
    --model_name_or_path ${model_name} \
    --data ${data_name} \
    --output_dir ${output_dir} \
    --per_gpu_train_batch_size ${batch_size} \
    --per_gpu_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${lr} \
    --epochs ${epochs} \
    --max_length ${max_length} \
    --max_num_instances_flan ${max_num_instances_flan} \
    --max_num_instances_ie ${max_num_instances_ie} \
    --add_input_features ${add_input_features} \
    --add_output_features ${add_output_features} \
    --add_label_space ${add_label_space} \
    --shuffle ${shuffle} \
    --text_version ${text_version} \
    --peft_type lora \
    --lora_target_modules ${lora_target_modules} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    >${stdout_file} 2>${stderr_file}


if [[ ${verbose} != True ]]
then
  tail -n 200 ${output_dir}/run.log
fi
