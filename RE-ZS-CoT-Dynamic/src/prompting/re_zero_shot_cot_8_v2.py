from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class REZSCOT8V2(Prompting):
    """(Variant 2)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        magic = 'You are an expert Polymath. Use your knowledge as an expert Polymath to think step-by-step before answering.\n'
        return  prompt + magic 

    def __str__(self) -> str:
        return "Role-Enhanced Zero-Shot Chain-of-Thought Prompting (Polymath)"
