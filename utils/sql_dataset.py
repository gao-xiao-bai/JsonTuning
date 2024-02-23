import json
import os
import random
import datasets
import re
import copy
from utils.templates import NL2SQL_templates


logger = datasets.logging.get_logger(__name__)


def get_input_text(format_string, feature_dictionary, schema_text):
    to_join = []
    parts = [p for p in re.split(r"({[\w\s]*})", format_string) if p]
    for part in parts:
        if part[0] == "{" and part[-1] == "}":
            if part[1:-1] == "database schema":
                t = schema_text
            else:
                t = feature_dictionary[part[1:-1]]
            to_join.append(t)
        else:
            to_join.append(part)
    text = "".join(to_join)
    return text


def assign_template(x, template):
    x_new = copy.deepcopy(x)
    x_new["template"] = template
    x_new["input_text"] = get_input_text(template[0], x["input_json"], x["schema_text"])
    return x_new


def construct_test_data(data, use_all_templates=0):
    constructed_data = []
    for x in data:
        task_name = x["task name"]
        templates = NL2SQL_templates

        if not use_all_templates:
            templates = templates[0:1]

        for template in templates:
            x_new = assign_template(x, template)
            constructed_data.append(x_new)

    return constructed_data


class Config(datasets.BuilderConfig):
    def __init__(self, *args, data_path=None, max_num_instances=None, use_all_templates=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path: str = data_path
        self.max_num_instances: int = max_num_instances
        self.use_all_templates: int = use_all_templates


class JsonSQL(datasets.GeneratorBasedBuilder):

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
                    "query": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "db_id": datasets.Value("string"),
                    "db_path": datasets.Value("string"),
                    "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                    "db_column_names": datasets.features.Sequence(
                        {
                            "table_id": datasets.Value("int32"),
                            "column_name": datasets.Value("string"),
                        }
                    ),
                    "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                    "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                    "db_foreign_keys": datasets.features.Sequence(
                        {
                            "column_id": datasets.Value("int32"),
                            "other_column_id": datasets.Value("int32"),
                        }
                    ),
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
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)[:max_num_instances]
                data = construct_test_data(data, self.config.use_all_templates)
                for i, example in enumerate(data):
                    task_name = example["task name"]
                    task_source = example["task source"]
                    instruction = " ".join(example["template"])
                    example["input_json"]["instruction"] = instruction

                    json_input = {}
                    json_input["input"] = example["input_json"]
                    json_input["output features"] = example["output features"]
                    json_output = example["output_json"]

                    data_point = {}
                    data_point["question"] = example["question"]
                    data_point["query"] = example["query"]
                    data_point["db_id"] = example["db_id"]
                    data_point["db_path"] = example["db_path"]
                    data_point["db_table_names"] = example["db_table_names"]
                    data_point["db_column_names"] = example["db_column_names"]
                    data_point["db_column_types"] = example["db_column_types"]
                    data_point["db_primary_keys"] = example["db_primary_keys"]
                    data_point["db_foreign_keys"] = example["db_foreign_keys"]
                    data_point["text_input"] = example["input_text"]
                    data_point["text_output"] = example["output_text"]
                    data_point["json_input"] = json.dumps(json_input)
                    data_point["json_output"] = json.dumps(json_output)
                    data_point["template"] = " ".join(example["template"])
                    data_point["task name"] = task_name
                    data_point["task source"] = task_source

                    yield f"{task_source}_{task_name}_{i}", data_point

        else:
            logger.info(f"File {path} does not exist. Ignore ...")
