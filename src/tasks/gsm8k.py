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


class GSM8K(Task, ABC):
    """GSM8K"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'GSM8K.jsonl'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'GSM8K_train.jsonl'

    def __new__(cls, *args, **kwargs):
        if cls is GSM8K:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return True

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        dev_list = cls.get_task_list(cls.dev_dataset_path)
        few_shot_examples = random.sample(dev_list, NUM_FEW_SHOT_SAMPLES)
        return few_shot_examples

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        special_token = "### "
        answer_key = item["answer"]
        answer_key = answer_key[answer_key.find(special_token) + len(special_token):]

        calculator_annotation_pattern = re.compile(r'<<.*?>>')
        reason = re.sub(calculator_annotation_pattern, "", item["answer"])

        return DataItem(f"Question: {item['question']}\nAnswer:", reason,
                        answer_key)

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
        # Patterns to match final answer in different formats:
        # This pattern looks for a currency value, e.g. "$70,000.00"
        currency_pattern = r"\$(\d[\d,]*\.?\d*)"
        # This pattern matches explicit answer declarations like "Answer: 42" (also handles commas)
        secondary_pattern = r"(?:answer|result|solution)[:\s]*(\d[\d,]*\.?\d*)\b"
        # This pattern matches standalone numbers (also handles commas)
        primary_pattern = r"\b\d[\d,]*\.?\d*\b"
    
        lines = response.splitlines()
        prediction = None
    
        # Search all lines in reverse order (last lines first)
        for line in reversed(lines):
            line = line.strip()
            # 1. Check for explicit answer declarations first
            answer_match = re.search(secondary_pattern, line, re.IGNORECASE)
            if answer_match:
                prediction = answer_match.group(1)
                break
            # 2. Check for currency amounts if no explicit declaration is found
            currency_match = re.search(currency_pattern, line)
            if currency_match:
                prediction = currency_match.group(1)
                break
            # 3. Fallback: check for any standalone number
            numbers = re.findall(primary_pattern, line)
            if numbers:
                prediction = numbers[-1]  # Take the last number in the line
                break
    
        if prediction:
            # Remove commas (e.g., turn "70,000" into "70000")
            prediction_clean = prediction.replace(',', '')
            answer_clean = answer.replace(',', '')
            try:
                prediction_val = float(prediction_clean)
                answer_val = float(answer_clean)
            except ValueError:
                prediction_val = prediction_clean
                answer_val = answer_clean
            logger.debug(f"Prediction: {prediction_val}, Answer: {answer_val}")
            return prediction_val == answer_val, prediction
        logger.debug("Could not extract prediction from response")
        return False, ""




    def __str__(self) -> str:
        return "GSM8K"
