import json
import os
import random
import datasets
import re
from .templates import t0_question_answer


logger = datasets.logging.get_logger(__name__)


class Config(datasets.BuilderConfig):
    def __init__(self, *args, data_path=None, max_num_instances=None, use_all_templates=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path: str = data_path
        self.max_num_instances: int = max_num_instances
        self.use_all_templates: int = use_all_templates


class JsonBBH(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = Config
    BUILDER_CONFIGS = [
        Config(name="default", description="Default config")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "text_input": datasets.Value("string"),
                    "text_output": datasets.Value("string"),
                    "json_input": datasets.Value("string"),
                    "json_output": datasets.Value("string"),
                    "template": datasets.Value("string"),
                    "task name": datasets.Value("string"),
                    "task source": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        data_path = self.config.data_path

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": data_path,
                    "max_num_instances": self.config.max_num_instances,
                }),
        ]

    def _generate_examples(self, path=None, max_num_instances=None):
        """Yields examples."""
        logger.info(f"Generating data from = {path}")

        def get_input_text(input_json, template):
            to_join = []
            parts = [p for p in re.split(r"({\w*})", template) if p]
            for part in parts:
                if part[0] == "{" and part[-1] == "}":
                  t = input_json[part[1:-1]]
                  if type(t) is not str:
                      t = json.dumps(t)
                  to_join.append(t)
                else:
                  to_join.append(part)
            text = "".join(to_join)
            return text

        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if max_num_instances != -1:
                random.seed(0)
                random.shuffle(data)
                data = data[:max_num_instances]

            if self.config.use_all_templates:
                templates = t0_question_answer
            else:
                templates = t0_question_answer[3:4]

            for template_id, template in enumerate(templates):
                for i, example in enumerate(data):
                    task_name = example["task name"]
                    task_source = example["task source"]
                    instruction = " ".join(template)
                    example["input_json"]["instruction"] = instruction
                    example["input_text"] = get_input_text(example["input_json"], template[0])

                    json_input = {}
                    json_input["input"] = example["input_json"]
                    json_input["output features"] = example["output features"]
                    json_output = example["output_json"]

                    data_point = {}
                    data_point["text_input"] = example["input_text"]
                    data_point["text_output"] = example["output_text"]
                    data_point["json_input"] = json.dumps(json_input)
                    data_point["json_output"] = json.dumps(json_output)
                    data_point["template"] = " ".join(template)
                    data_point["task name"] = task_name
                    data_point["task source"] = task_source

                    yield f"{template_id}_{task_source}_{task_name}_{i}", data_point

        else:
            logger.info(f"File {path} does not exist. Ignore ...")
