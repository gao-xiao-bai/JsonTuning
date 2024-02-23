import logging
import random
import string
import copy
import json
import torch
import transformers
from transformers.data.data_collator import *
from typing import Dict, Optional, Sequence, Union, Any

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_length: int, padding: Union[bool, str, PaddingStrategy] = "longest") -> Dict:
    """Tokenize a list of strings."""
    tokenized_strings = tokenizer(
        strings,
        return_tensors="pt",
        padding=padding,
        max_length=max_length,
        truncation=True,
    )
    input_ids = labels = tokenized_strings.input_ids
    input_ids_lens = labels_lens = [
        x.ne(tokenizer.pad_token_id).sum().item() for x in input_ids
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
    padding: Union[bool, str, PaddingStrategy] = "longest"
) -> tuple:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, max_length, padding) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len, source in zip(labels, sources_tokenized["input_ids_lens"], sources_tokenized["input_ids"]):
        label[:source_len] = IGNORE_INDEX
    labels = labels + (labels == tokenizer.pad_token_id) * IGNORE_INDEX
    return input_ids, labels


def preprocess_eval(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
    padding: Union[bool, str, PaddingStrategy] = "longest"
) -> tuple:
    """Preprocess the data by tokenizing."""
    targets_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, max_length, padding) for strings in (targets, sources)]
    input_ids = sources_tokenized["input_ids"]
    labels = targets_tokenized["input_ids"]
    return input_ids, labels



@dataclass
class JsonDataCollator:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    add_output_control: bool = True
    add_label_space: bool = True

    def __call__(self, batch, eval=False):

        self.tokenizer.padding_side = "right" if not eval else "left"

        sources = []
        targets = []
        for instance in batch:
            model_input = json.loads(instance["json_input"])

            if not self.add_output_control:
                model_input.pop("output features")

            if not self.add_label_space:
                if "candidate answers" in model_input["input"]:
                    model_input["input"].pop("candidate answers")
                if "candidate outputs" in model_input["input"]:
                    model_input["input"].pop("candidate outputs")
                if self.add_output_control:
                    if "answer" in model_input["output features"] and "description" in model_input["output features"]["answer"]:
                        model_input["output features"]["answer"].pop("description")
                    if "output" in model_input["output features"] and "description" in model_input["output features"]["output"]:
                        model_input["output features"]["output"].pop("description")

            source = json.dumps(model_input)
            source = source + self.tokenizer.bos_token
            sources.append(source)

            target = f"{instance['json_output']}{self.tokenizer.eos_token}"
            targets.append(target)

        if not eval:
            input_ids, labels = preprocess(sources, targets, self.tokenizer, self.max_length, self.padding)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
        else:
            input_ids, labels = preprocess_eval(sources, targets, self.tokenizer, self.max_length, self.padding)
            return dict(
                input_ids=input_ids,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )


@dataclass
class TextDataCollator:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None

    def __call__(self, batch, eval=False):

        self.tokenizer.padding_side = "right" if not eval else "left"

        sources = []
        targets = []
        for instance in batch:
            source = instance["text_input"] + "\n"
            target = instance["text_output"] + self.tokenizer.eos_token

            sources.append(source)
            targets.append(target)

        if not eval:
            input_ids, labels = preprocess(sources, targets, self.tokenizer, self.max_length, self.padding)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
        else:
            input_ids, labels = preprocess_eval(sources, targets, self.tokenizer, self.max_length, self.padding)
            return dict(
                input_ids=input_ids,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

