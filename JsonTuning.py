import os
import bitsandbytes as bnb
import logging
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
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

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


logger = logging.getLogger(__name__)


def get_data_model(args, output_dir):

    DATA_PATH = {
        "flan": ["./jc-data/flan2022_50K.json", "utils/flan_dataset.py", args.max_num_instances_flan],
        "ie": ["./jc-data/ie_5K.json", "utils/ie_dataset.py", args.max_num_instances_ie],
        "alpaca": ["tatsu-lab/alpaca", "utils/alpaca_dataset.py", args.max_num_instances_alpaca],
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

        if max_num_instances == 0:
            continue

        datasets = load_dataset(
            load_script,
            data_path=data_path,
            cache_dir=args.cache_dir,
            max_num_instances=max_num_instances,
        )

        print(datasets)

        all_datasets.append(datasets)

    raw_datasets = all_datasets[0]
    raw_datasets["train"] = concatenate_datasets([x["train"] for x in all_datasets])
    print(raw_datasets)

    model_class = get_model_class(args.model_type)
    peft_class = get_peft_class(args.peft_type)

    model = model_class.model.from_pretrained(args.model_name_or_path,
                                            cache_dir=args.cache_dir,
                                            load_in_8bit=True,
                                            device_map=device_map,
                                            trust_remote_code=True)
    model = prepare_model_for_int8_training(model)
    print(model)

    tokenizer = model_class.tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    if args.model_type in ['llama']:
        tokenizer.pad_token_id = 0  # unk_id in llama
    if args.model_type in ['falcon']:
        tokenizer.bos_token_id = 8  # >>PREFIX<<
        tokenizer.pad_token_id = 9  # >>SUFFIX<<

    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
            )

    if args.peft_type == 'lora':
        config = peft_class(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        assert args.peft_type, "Error: Wrong type of peft."

    model = get_peft_model(model, config)

    # the size of trainable parameters for lora modules
    model.print_trainable_parameters()

    return raw_datasets, model, tokenizer


def train(args):

    # Set seed
    set_seed(args.seed)

    output_dir = args.output_dir

    # load data & model_class
    raw_datasets, model, tokenizer = get_data_model(args, output_dir)

    if args.local_rank <= 0:
        import wandb

        init_args = {}
        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        # wandb.init(
        #     project="JsonTuning",
        #     name=output_dir,
        #     entity=YOUR_USER_NAME,
        #     **init_args,
        # )
        # wandb.config.update(args, allow_val_change=True)

    train_data = raw_datasets["train"]

    if args.text_tuning:
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

    # train
    total_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps * (world_size if ddp else 1)
    total_optim_steps = train_data.num_rows // total_batch_size
    saving_step = int(total_optim_steps / 10)
    warmup_steps = int(total_optim_steps / 10)

    print("***** Running training *****")
    print(f"  Num Epochs = {args.epochs}", )
    print(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Total optimization steps = {total_optim_steps}")
    print(f"  Saving steps = {saving_step}")

    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    trainer = JsonTrainer(
        model=model,
        train_dataset=train_data,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=args.per_gpu_train_batch_size,
            per_device_eval_batch_size=args.per_gpu_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=saving_step,
            save_steps=saving_step,
            output_dir=output_dir,
            save_total_limit=11,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            remove_unused_columns=False,
        ),
        data_collator=data_collator,
        callbacks=[SavePeftModelCallback],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":

    # model arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--size', type=str, help='the size of llama model')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--model_type', default="llama", choices=['llama', 'falcon'])
    parser.add_argument('--model_name_or_path', default="yahma/llama-7b-hf", type=str)
    parser.add_argument('--per_gpu_train_batch_size', default=4, type=int, help='Batch size per GPU for training.')
    parser.add_argument('--per_gpu_eval_batch_size', default=4, type=int, help='Batch size per GPU for evaluation.')
    parser.add_argument('--gradient_accumulation_steps', default=16, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--cache_dir', default="./cache", type=str)
    parser.add_argument('--seed', default=42, type=int)

    # data arguments
    parser.add_argument('--output_dir', type=str, help='')
    parser.add_argument('--data', type=str, help='the data used for instructing tuning')
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--max_num_instances_flan', default=50000, type=int)
    parser.add_argument('--max_num_instances_ie', default=5000, type=int)
    parser.add_argument('--max_num_instances_alpaca', default=-1, type=int)
    parser.add_argument('--add_output_control', default=1, type=int)  
    parser.add_argument('--add_label_space', default=1, type=int) 
    parser.add_argument('--text_tuning', default=0, type=int) 

    # PEFT arguments
    parser.add_argument('--peft_type', default="lora")
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--lora_target_modules', default="q_proj,v_proj", type=str,
                        help="the module to be injected, e.g. q_proj/v_proj/k_proj/o_proj for llama, query_key_value for falcon",)
    parser.add_argument('--resume_from_checkpoint', nargs='?', default=None, const=True, help='resume from the specified or the latest checkpoint, e.g. `--resume_from_checkpoint [path]` or `--resume_from_checkpoint`')

    args, _ = parser.parse_known_args()
    print(args)

    train(args)
