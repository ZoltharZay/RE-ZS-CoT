from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class REZSCOT2V1(Prompting):
    """(Variant 1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        magic = 'You are an expert Physicist. Use your knowledge as an expert Physicist to think step-by-step before answering.\n'
        return  magic + prompt 

    def __str__(self) -> str:
        return "Role-Enhanced Zero-Shot Chain-of-Thought Prompting (Physicist)"
