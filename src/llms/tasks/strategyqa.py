import json
import os
import random
import re
from abc import ABC
from pathlib import Path

from loguru import logger

from src.config import DATASETS_DIRECTORY, NUM_FEW_SHOT_SAMPLES
from src.models.data_item import DataItem
from src.tasks.task import Task


class StrategyQA(Task, ABC):
    """StrategyQA"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'StrategyQA.json'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'StrategyQA_train.json'

    def __new__(cls, *args, **kwargs):
        if cls is StrategyQA:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return True

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        dev_list = []
        with open(cls.dev_dataset_path) as dataset:
            data = json.load(dataset)
            for item in data:
                parsed_item = DataItem(f"Question: {item['question']}\nChoices: A) True, B) False\nAnswer:",
                                       " ".join(item["facts"]),
                                       "A" if item["answer"] is True else "B")
                dev_list.append(parsed_item)
        few_shot_examples = random.sample(dev_list, NUM_FEW_SHOT_SAMPLES)
        return few_shot_examples

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        choices = "A) True, B) False"
        return DataItem(f"Question: {item['input']}\nChoices: {choices}\nAnswer:", item["target"],
                        "A" if item["target_scores"]["Yes"] == 1 else "B")

    @classmethod
    def get_task_list(cls, data_path=dataset_path) -> list[DataItem]:
        task_list = []
        with open(data_path) as dataset:
            data = json.load(dataset)
            for item in data['examples']:
                parsed_item = cls.get_task(item)
                task_list.append(parsed_item)
        return task_list

    @classmethod
    def evaluate(cls, response: str, answer: str) -> (bool, str):
        # Patterns to match different answer formats
        primary_pattern = r"\b([AB])\)|(True|False)\b"  # Matches "A)", "B)", "True", or "False"
        secondary_pattern = r"(?:answer|correct|solution)[:\s-]+([ABTtFf])[\)\.]?"  # Matches "Answer: B", "Correct - A", etc.

        lines = response.splitlines()
        prediction = None

        # Search lines from bottom up (prioritize ending)
        for line in reversed(lines):
            line = line.strip().upper()
            
            # Try secondary pattern first (explicit answer declarations)
            sec_match = re.search(secondary_pattern, line, re.IGNORECASE)
            if sec_match:
                raw_pred = sec_match.group(1).upper()
                prediction = 'A' if raw_pred in ['A', 'T'] else 'B'
                break
                
            # Try primary pattern (direct answer formats)
            pri_match = re.search(primary_pattern, line, re.IGNORECASE)
            if pri_match:
                if pri_match.group(1):  # A/B format
                    prediction = pri_match.group(1).upper()
                else:  # True/False format
                    prediction = 'A' if pri_match.group(2).upper() == 'TRUE' else 'B'
                break

        if prediction:
            logger.debug(f"Prediction: {prediction}, Answer: {answer}")
            return prediction == answer, prediction
        logger.debug(f"Could not extract prediction from response")
        return False, ""

    def __str__(self) -> str:
        return "StrategyQA"
