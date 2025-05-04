import os
import re
from abc import ABC
from pathlib import Path

from jsonlines import jsonlines
from loguru import logger

from src.config import DATASETS_DIRECTORY
from src.models.data_item import DataItem
from src.tasks.task import Task


class Winogrande(Task, ABC):
    """Winogrande"""

    dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'Winogrande.jsonl'
    dev_dataset_path = Path(os.getcwd()) / DATASETS_DIRECTORY / 'Winogrande_train.jsonl'

    def __new__(cls, *args, **kwargs):
        if cls is Winogrande:
            raise TypeError(f"'{cls.__name__}' cannot be instantiated")
        return object.__new__(cls)

    @classmethod
    def has_native_cot_samples_supported(cls) -> bool:
        return False

    @classmethod
    def get_few_shot_samples(cls) -> list[DataItem]:
        few_shot_examples = [
            DataItem(
                "Question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine.\nChoices: 1) Ian, 2) Dennis\nAnswer:",
                "Ian volunteered to eat Dennis's menudo, implying that it was Dennis who despised eating intestine.",
                "2"),
            DataItem(
                "Question: Ian volunteered to eat Dennis's menudo after already having a bowl because _ enjoyed eating intestine.\nChoices: 1) Ian, 2) Dennis\nAnswer:",
                "Ian volunteered to eat Dennis's menudo, implying that Dennis enjoys eating intestines and may already feel full.",
                "1"),
            DataItem(
                "The GPS and map helped me navigate home.  I got lost when the _ got turned upside down.\nChoices: 1) GPS, 2) map\nAnswer:",
                "Since map is an object, it is more likely to be turned upside down than GPS, which is a system.",
                "2"),
            DataItem(
                "The GPS and map helped me navigate home.  I got lost when the _ got turned off.\nChoices: 1) GPS, 2) map\nAnswer:",
                "Since GPS is a system, it is more likely to be turned off than map, which is an object.",
                "1"),
            DataItem(
                "Donald was richer than Leslie was because companies had found oil on the property of _ .\nChoices: 1) Donald, 2) Leslie\nAnswer:",
                "Since Donald is richer, it is more likely that oil was found on his property than on Leslie's.",
                "1"),
            DataItem(
                "Donald was poorer than Leslie was because companies had found oil on the property of _ .\nChoices: 1) Donald, 2) Leslie\nAnswer:",
                "Since Donald is poorer, it is less likely that oil was found on his property than on Leslie's.",
                "2"),
            DataItem(
                "Adam put handwash only clothes in the washer but Aaron washed them by hand as _ was lazy.\nChoices: 1) Adam, 2) Aaron\nAnswer:",
                "Since Aaron washed the clothes by hand, it is more likely that Adam was lazy.",
                "1"),
            DataItem(
                "Adam put handwash only clothes in the washer but Aaron washed them by hand as _ was conscientious.\nChoices: 1) Adam, 2) Aaron\nAnswer:",
                "Since Aaron washed the clothes by hand, it is less likely that Adam was conscientious.",
                "2"),
            DataItem(
                "The woman sprayed cleaner on the mirror but skipped the countertop because the _ was clean.\nChoices: 1) mirror, 2) countertop\nAnswer:",
                "Since the woman skipped the countertop, it is more likely that the countertop was clean.",
                "2"),
            DataItem(
                "The woman sprayed cleaner on the mirror but skipped the countertop because the _ was dirty.\nChoices: 1) mirror, 2) countertop\nAnswer:",
                "Since the woman skipped the countertop, it is less likely that the mirror was dirty.",
                "1"),
        ]
        return few_shot_examples

    @classmethod
    def get_task(cls, item: dict) -> DataItem:
        choices = f"1) {item['option1']}, 2) {item['option2']}"
        return DataItem(f"Question: {item['sentence']}\nChoices: {choices}\nAnswer:", None,
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
        # Patterns to match the answer in different formats
        primary_pattern = r"\b([1-2])\)?"  # Matches "1)", "2)", or "1", "2"
        secondary_pattern = r"(?:answer|correct answer)[:\s]*([1-2])\b"  # Matches "Answer: 1", "Correct answer is 2", etc.

        lines = response.splitlines()
        extracted_answer = None

        for line in lines:
            line = line.strip()
            # Check primary pattern (e.g., "2)" or "2")
            primary_match = re.search(primary_pattern, line)
            if primary_match:
                extracted_answer = primary_match.group(1)
                break
            # Check secondary pattern (e.g., "Answer: 1")
            secondary_match = re.search(secondary_pattern, line, re.IGNORECASE)
            if secondary_match:
                extracted_answer = secondary_match.group(1)
                break

        if extracted_answer is not None:
            logger.debug(f"Prediction: {extracted_answer}, Answer: {answer}")
            return extracted_answer == answer, extracted_answer
        logger.debug(f"Could not extract prediction from response")
        return False, ""

    def __str__(self) -> str:
        return "Winogrande"
