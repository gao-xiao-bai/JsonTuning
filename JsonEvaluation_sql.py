import os
import re
import json
import torch
import numpy as np
import bitsandbytes as bnb
from dataclasses import dataclass, field
from datasets import load_dataset, concatenate_datasets
import transformers
from collections import namedtuple

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast,
    GenerationConfig,
    set_seed,
)

from peft import (
    PeftModel,
    prepare_model_for_int8_training,
    AdaLoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

import argparse
from utils.collator import JsonDataCollator, TextDataCollator
from utils.compute_metrics_sql import compute_metrics, compute_grouped_metrics
from utils.trainer import JsonTrainer


device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

ModelClass = namedtuple("ModelClass", ('tokenizer', 'model'))

_MODEL_CLASSES = {
    "llama": ModelClass(**{
        "tokenizer": LlamaTokenizer,
        "model": LlamaForCausalLM,

    }),
    "Auto": ModelClass(**{
        "tokenizer": AutoTokenizer,
        "model": AutoModelForCausalLM,
    })
}
_PEFT_CLASSES = {
    "lora": LoraConfig,
}


def get_data_model(args):

    DATA_PATH = {
        "sql": ["./jc-data/sql.json", "utils/sql_dataset.py", args.max_num_instances_sql],
    }

    def get_model_class(model_type):

        if model_type not in ['llama']:
            model_type = "Auto"

        return _MODEL_CLASSES[model_type]  # tokenizer, model

    def get_peft_class(peft_type):

        return _PEFT_CLASSES[peft_type]

    all_datasets = []
    for data in args.data.split(","):
        data_path, load_script, max_num_instances = DATA_PATH.get(data, None)
        assert data_path, "Error: Wrong type of data."

        # Get the dataset
        datasets = load_dataset(
            load_script,
            data_path=data_path,
            cache_dir=args.cache_dir,
            max_num_instances=max_num_instances,
            use_all_templates=args.use_all_templates,
        )

        all_datasets.append(datasets)

    raw_datasets = all_datasets[0]
    raw_datasets["test"] = concatenate_datasets([x["test"] for x in all_datasets])
    print(raw_datasets)

    model_class = get_model_class(args.model_type)
    model = model_class.model.from_pretrained(args.model_name_or_path,
                                            cache_dir=args.cache_dir,
                                            load_in_8bit=True,
                                            device_map=device_map,
                                            trust_remote_code=True)

    tokenizer = model_class.tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    if args.model_type in ['llama']:
        tokenizer.pad_token_id = 0  # unk_id in llama.
    if args.model_type in ['falcon']:
        tokenizer.bos_token_id = 8
        tokenizer.pad_token_id = 9

    tokenizer.padding_side = "left"

    peft_model_id = args.resume_from_checkpoint
    peft_class = get_peft_class(args.peft_type)
    config = peft_class.from_pretrained(peft_model_id)
    model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16)
    print(model)

    return raw_datasets, model, tokenizer


def normalize(query: str) -> str:
    def bracket_op_fix(s):
        s = s.replace("(", " (")
        s = s.replace("( ", "(")
        s = s.replace(")", ") ")
        s = s.replace(" )", ")")
        s = s.replace("! =", "!=")
        s = s.replace("< =", "<=")
        s = s.replace("> =", ">=")
        return s

    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(
            r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s
        )

    return white_space_fix(bracket_op_fix(comma_fix(white_space_fix(lower(query)))))


def get_answer(output):
    return normalize(output)


def normalize_answer_text(s, input_text):
    output = s.replace(input_text, "").strip()

    return output, get_answer(output)


EVALUATION_KEY = "SQL query"


def repair(s):
    s = s.replace('{"SQL query"}', "")
    s = s + '"}'
    return s


def normalize_answer_json(s, input_text):
    evaluation_key = EVALUATION_KEY
    s = s.replace(input_text, "").strip()
    ini_output = s
    try:
        try:
            output = json.loads(s)
            output = output[evaluation_key]
        except:
            s = repair(s)
            output = json.loads(s)
            output = output[evaluation_key]
    except:
        output = ""

    if type(output) is not str:
        output = str(output)

    return ini_output, normalize(output)


def evaluate(args):

    # Set seed
    set_seed(args.seed)

    checkpoint_path = args.resume_from_checkpoint

    exp_name, parameters = checkpoint_path.split("/")[:2]
    if "json" in exp_name:
        args.text_tuning = 0
    else:
        args.text_tuning = 1

    if "llama2" in exp_name:
        if "13b" in exp_name:
            args.model_name_or_path = "meta-llama/Llama-2-13b-hf"
        else:
            args.model_name_or_path = "meta-llama/Llama-2-7b-hf"
    elif "llama" in exp_name:
        if "13b" in exp_name:
            args.model_name_or_path = "yahma/llama-13b-hf"
        else:
            args.model_name_or_path = "yahma/llama-7b-hf"
    elif "falcon" in exp_name:
        args.model_name_or_path = "tiiuae/falcon-7b"
    else:
        assert False, f"The exp_name {exp_name} is wrong!"

    parameters = parameters.split("_")
    for x in parameters:
        if x.startswith("ml"):
            args.max_length = int(x[2:6])
        if x.startswith("aoc"):
            args.add_output_control = int(x[3])
        if x.startswith("als"):
            args.add_label_space = int(x[3])

    # print("args: ", args)

    if not os.path.exists(checkpoint_path):
        assert False, f"The path {checkpoint_path} does not exist!"

    if args.local_rank <= 0:
        import wandb

        init_args = {}
        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        # wandb.init(
        #     project="JsonEval-SQL",
        #     name=f"{checkpoint_path}_uat{args.use_all_templates}",
        #     entity=YOUR_USER_NAME,
        #     **init_args,
        # )
        # wandb.config.update(args, allow_val_change=True)

    # 1. load data & model_class
    raw_datasets, model, tokenizer = get_data_model(args)

    # 2. Obtain dataset
    test_data = raw_datasets["test"]

    if args.text_tuning == 1:
        data_collator = TextDataCollator(
            tokenizer,
            model=model,
            max_length=args.max_length,
        )
    else:
        data_collator = JsonDataCollator(
            tokenizer,
            model=model,
            max_length=args.max_length,
            add_output_control=args.add_output_control,
            add_label_space=args.add_label_space,
        )

    def compute_sql_metrics(dataset, inputs, preds, save_prefix=None):
        task_names = [x["task name"] for x in dataset]
        categories = [x["template"] for x in dataset]
        for i, input in enumerate(inputs):
            inputs[i] = input + (input == -100) * 100
        for i, pred in enumerate(preds):
            preds[i] = pred + (pred == -100) * 100

        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        normalize_func = normalize_answer_json if not args.text_tuning else normalize_answer_text
        decoded_preds = [normalize_func(x, decoded_inputs[i]) for i, x in enumerate(decoded_preds)]
        ini_decoded_preds = [x[0] for x in decoded_preds]
        decoded_preds = [x[1] for x in decoded_preds]
        references = [x for i, x in enumerate(dataset)]

        if save_prefix is not None:
            with open(os.path.join(checkpoint_path, f"{save_prefix}_{args.use_all_templates}_sql_predictions.jsonl"), "w") as fout:
                for example, ini_pred, pred in zip(dataset, ini_decoded_preds, decoded_preds):
                    json_input = json.loads(example["json_input"])
                    json_output = json.loads(example["json_output"])
                    fout.write(json.dumps({
                        "task": example["task name"],
                        "json_input": json_input,
                        "json_output": json_output,
                        "initial_prediction": ini_pred,
                        "prediction": pred
                    }) + "\n")
            wandb.save(os.path.join(checkpoint_path, f"{save_prefix}_{args.use_all_templates}_sql_predictions.jsonl"))

        result = {}
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=task_names)
        result.update(result_per_task)
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories)
        result.update(result_per_category)

        exact_match_result_per_category = {x: y for x, y in result_per_category.items() if "exact_match" in x}
        exact_match_results = list(exact_match_result_per_category.values())
        exact_match_mean = np.mean(exact_match_results)
        exact_match_std = np.std(exact_match_results)
        result.update({
            "exact_match_mean": exact_match_mean,
            "exact_match_std": exact_match_std,
        })
        
        exec_result_per_category = {x: y for x, y in result_per_category.items() if "exec" in x}
        exec_results = list(exec_result_per_category.values())
        exec_mean = np.mean(exec_results)
        exec_std = np.std(exec_results)
        result.update({
            "exec_mean": exec_mean,
            "exec_std": exec_std,
        })

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        with open(os.path.join(checkpoint_path, f"{save_prefix}_{args.use_all_templates}_sql_results.json"), "w") as fout:
            json.dump(result, fout, indent=4)
        wandb.save(os.path.join(checkpoint_path, f"{save_prefix}_{args.use_all_templates}_sql_results.json"))

        return result

    # 3. Eval
    print("***** Running Evaluation *****")
    print(f"  Instantaneous batch size per GPU = {args.per_gpu_eval_batch_size}")

    generation_config = GenerationConfig(
        top_p=None,
        top_k=None,
        num_beams=1,
        temperature=0,
        do_sample=False,
        max_length=args.max_length // 2,
        max_new_tokens=200,
        output_scores=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    trainer = JsonTrainer(
        model=model,
        tokenizer=tokenizer,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_eval_batch_size=args.per_gpu_eval_batch_size,
            fp16=True,
            logging_steps=20,
            output_dir=checkpoint_path,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            remove_unused_columns=False,
            generation_config=generation_config,
            eval_accumulation_steps=10,
            predict_with_generate=True,
            include_inputs_for_metrics=True,
        ),
        data_collator=data_collator,
        compute_metrics=compute_sql_metrics,
    )

    test_results = trainer.evaluate(
        test_data, metric_key_prefix="test",
    )

    trainer.log(test_results)
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)


if __name__ == "__main__":

    # model arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--size', type=str, help='the size of llama model')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--model_type', default="llama", choices=['llama', 'falcon'])
    parser.add_argument('--model_name_or_path', default="yahma/llama-7b-hf", type=str)
    parser.add_argument('--per_gpu_eval_batch_size', default=4, type=int, help='Batch size per GPU for evaluation.')
    parser.add_argument('--cache_dir', default="./cache", type=str)
    parser.add_argument('--seed', default=42, type=int)

    # data arguments
    parser.add_argument('--data', type=str, help='the data used for instructing tuning')
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--max_num_instances_sql', default=-1, type=int)
    parser.add_argument('--use_all_templates', default=0, type=int)
    parser.add_argument('--add_output_control', default=1, type=int)
    parser.add_argument('--add_label_space', default=1, type=int)
    parser.add_argument('--text_tuning', default=0, type=int)

    # PEFT arguments
    parser.add_argument('--peft_type', default="lora", choices=['lora'])
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--lora_target_modules', default="q_proj,v_proj", type=str,
                        help="the module to be injected, e.g. q_proj/v_proj/k_proj/o_proj for llama, query_key_value for falcon")
    parser.add_argument('--resume_from_checkpoint', nargs='?', default=None, const=True, help='resume from the specified or the latest checkpoint, e.g. `--resume_from_checkpoint [path]` or `--resume_from_checkpoint`')

    args, _ = parser.parse_known_args()

    evaluate(args)
