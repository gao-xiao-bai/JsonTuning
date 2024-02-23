import json
import os
import random
import datasets

logger = datasets.logging.get_logger(__name__)

class Config(datasets.BuilderConfig):
    def __init__(self, *args, data_path=None, max_num_instances=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path: str = data_path
        self.max_num_instances: int = max_num_instances


class JsonIE(datasets.GeneratorBasedBuilder):
    """JsonCollection."""

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
                })
        ]

    def _generate_examples(self, path=None, max_num_instances=None):
        """Yields examples."""
        logger.info(f"Generating data from = {path}")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for task in data.keys():
                task_data = data[task][:max_num_instances]
                for i, example in enumerate(task_data):
                    task_name = example["task name"]
                    task_source = example["task source"]
                    instruction = " ".join(example["template"])
                    example["input_json"]["instruction"] = instruction

                    json_input = {}
                    json_input["input"] = example["input_json"]
                    json_input["output features"] = example["output features"]
                    json_output = example["output_json"]

                    data_point = {}
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
