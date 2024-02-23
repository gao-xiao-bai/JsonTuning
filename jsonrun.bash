#!/usr/bin/env bash
# -*- coding:utf-8 -*-


model=$1
size=$2
text_tuning=$3
device=$4

. config/${model}_${size}_${text_tuning}.ini && bash run_exp.bash run.bash ${device}


