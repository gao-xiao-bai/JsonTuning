#!/usr/bin/env bash
# -*- coding:utf-8 -*-

echo "Start Running ..."

script=$1
device=$2

read -ra GRADIENT_ACCUMULATION_STEPS <<<"${GRADIENT_ACCUMULATION_STEPS}"
read -ra LR_RATE <<<"${LR_RATE}"
read -ra EPOCHS <<<"${EPOCHS}"
read -ra LORA_TARGET_MODULES <<<"${LORA_TARGET_MODULES}"
read -ra LORA_R <<<"${LORA_R}"
read -ra LORA_ALPHA <<<"${LORA_ALPHA}"
read -ra LORA_DROPOUT <<<"${LORA_DROPOUT}"
read -ra MAX_LENGTH <<<"${MAX_LENGTH}"
read -ra MAX_NUM_INSTANCES_FLAN <<<"${MAX_NUM_INSTANCES_FLAN}"
read -ra MAX_NUM_INSTANCES_IE <<<"${MAX_NUM_INSTANCES_IE}"
read -ra DATA_NAME <<<"${DATA_NAME}"
read -r model_name <<<"${model_name}"
read -r model_type <<<"${model_type}"
read -r exp_name <<<"${exp_name}"
read -r batch_size <<<"${batch_size}"
read -r text_tuning <<<"${text_tuning}"

echo "model_name: " ${model_name}
echo "exp_name: " ${exp_name}
echo "device: " ${device}

for data_name in "${DATA_NAME[@]}"; do
  echo "data_name " ${data_name}
  for learning_rate in "${LR_RATE[@]}"; do
    echo "learning rate " ${learning_rate}
    for gradient_accumulation_steps in "${GRADIENT_ACCUMULATION_STEPS[@]}"; do
      echo "batch size " $((batch_size * gradient_accumulation_steps))
      for epochs in "${EPOCHS[@]}"; do
        echo "epochs " ${epochs}
        for max_length in "${MAX_LENGTH[@]}"; do
          echo "max_length " ${max_length}
          for lora_target_modules in "${LORA_TARGET_MODULES[@]}"; do
            echo "lora_target_modules " ${lora_target_modules}
            for lora_r in "${LORA_R[@]}"; do
              echo "lora_r " ${lora_r}
              for lora_alpha in "${LORA_ALPHA[@]}"; do
                echo "lora_alpha " ${lora_alpha}
                for lora_dropout in "${LORA_DROPOUT[@]}"; do
                  echo "lora_dropout " ${lora_dropout}
                  for max_num_instances_flan in "${MAX_NUM_INSTANCES_FLAN[@]}"; do
                    echo "max_num_instances_flan " ${max_num_instances_flan}
                    for max_num_instances_ie in "${MAX_NUM_INSTANCES_IE[@]}"; do
                      echo "max_num_instances_ie " ${max_num_instances_ie}

                      bash ${script}  \
                        -m ${model_name} \
                        -i ${data_name} \
                        --model_type ${model_type} \
                        --epochs ${epochs} \
                        --device ${device} \
                        --gradient_accumulation_steps ${gradient_accumulation_steps} \
                        --exp_name ${exp_name} \
                        --text_tuning ${text_tuning} \
                        --lora_target_modules ${lora_target_modules} \
                        --lora_r ${lora_r} \
                        --lora_alpha ${lora_alpha} \
                        --lora_dropout ${lora_dropout} \
                        --batch ${batch_size} \
                        --max_num_instances_flan ${max_num_instances_flan} \
                        --max_num_instances_ie ${max_num_instances_ie} \
                        --lr ${learning_rate} \
                        --max_length ${max_length} \

                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

exit 0
