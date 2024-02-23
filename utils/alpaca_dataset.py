import json
import random
import datasets
import re

logger = datasets.logging.get_logger(__name__)


def get_input_text(format_string, feature_dictionary):
    to_join = []
    parts = [p for p in re.split(r"({[\w\s]*})", format_string) if p]
    for part in parts:
        if part[0] == "{" and part[-1] == "}":
            t = feature_dictionary[part[1:-1]]
            if type(t) is list and type(t[0]) is str:
                t = ", ".join(t)
            elif type(t) is not str:
                t = json.dumps(t)
            to_join.append(t)
        else:
            to_join.append(part)
    text = "".join(to_join)
    return text


class Config(datasets.BuilderConfig):
    def __init__(self, *args, data_path=None, max_num_instances=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path: str = data_path
        self.max_num_instances: int = max_num_instances


class JsonAlpaca(datasets.GeneratorBasedBuilder):

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
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": data_path,
                    "max_num_instances": self.config.max_num_instances,
                }),
        ]

    def _generate_examples(self, path=None, max_num_instances=None):
        """Yields examples."""
        logger.info(f"Generating data from = {path}")

        if path:
            data = datasets.load_dataset(path, split="train", cache_dir="./cache")
            for i, example in enumerate(data):
                task_name = "alpaca"
                task_source = "standford-alpaca"
                template = ("Q:{question}\nA:", "{answer}")
                instruction = " ".join(template)

                question = " ".join([example["instruction"], example["input"]]) if example["input"] else example["instruction"]
                answer = example["output"]

                input_json = {"question": question}
                input_json["instruction"] = instruction

                json_input = {}
                json_input["input"] = input_json
                json_input["output features"] = {
                    "answer": {"type": "string"}
                }
                json_output = {
                    "answer": answer
                }

                data_point = {}
                data_point["text_input"] = get_input_text(template[0], input_json)
                data_point["text_output"] = answer
                data_point["json_input"] = json.dumps(json_input)
                data_point["json_output"] = json.dumps(json_output)
                data_point["template"] = " ".join(template)
                data_point["task name"] = task_name
                data_point["task source"] = task_source

                yield f"{task_source}_{task_name}_{i}", data_point

        else:
            logger.info(f"File {path} does not exist. Ignore ...")
