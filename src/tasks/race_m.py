import os
import random
import re
from abc import ABC
from pathlib import Path

from jsonlines import jsonlines
from loguru import logger

from src.config import DATASETS_DIRECTORY, NUM_FEW_SHOT_SAMPLES
from src.models.data_item import DataItem
from src.tasks.task import Task


class RACEMiddle(Task, ABC):
    """RACE-Middle"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'RACE_m.jsonl'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'RACE_m_dev.jsonl'

    def __new__(cls, *args, **kwargs):
        if cls is RACEMiddle:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return False

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        dev_list = cls.get_task_list(cls.dev_dataset_path)
        few_shot_examples = random.sample(dev_list, NUM_FEW_SHOT_SAMPLES)
        return few_shot_examples

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        choice_prefixes = ["A)", "B)", "C)", "D)"]
        choices = ", ".join([f"{prefix} {choice}" for prefix, choice in zip(choice_prefixes, item["options"])])
        return DataItem(f"Article: {item['article']}\nQuestion: {item['question']}\nChoices: {choices}\nAnswer:", None,
                        item["answer"])

    @classmethod
    def get_task_list(cls, data_path=dataset_path) -> list[DataItem]:
        task_list = []
        with jsonlines.open(data_path) as dataset:
            for item in dataset:
                parsed_item = cls.get_task(item)
                task_list.append(parsed_item)
        return task_list

    @classmethod
    def evaluate(cls, response: str, answer: str) -> (bool, str):
        # Patterns to match different answer formats
        primary_pattern = r"\b([A-D])\)"  # Matches "A)" style
        secondary_pattern = r"(?:answer|correct option)[:\s]*([A-D])\b"  # Matches "Answer: B" style

        lines = response.splitlines()
        prediction = None

        # Search all lines from bottom up
        for line in reversed(lines):
            line = line.strip().upper()
            # Check for letter+parenthesis format
            direct_match = re.search(primary_pattern, line)
            if direct_match:
                prediction = direct_match.group(1)
                break
            # Check annotated answers
            annotated_match = re.search(secondary_pattern, line)
            if annotated_match:
                prediction = annotated_match.group(1)
                break

        if prediction:
            logger.debug(f"Prediction: {prediction}, Answer: {answer}")
            return prediction == answer, prediction
        logger.debug(f"Could not extract prediction from response")
        return False, ""

    def __str__(self) -> str:
        return "RACE-Middle"
