from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class REZSCOT7V3(Prompting):
    """(Variant 3)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None) -> str:
        magic1 = 'You are an expert Software Developer.\n'
        magic2 = ' Use your knowledge as an expert Software Developer to think step-by-step before answering.\n'
        return  magic1 + prompt + magic2

    def __str__(self) -> str:
        return "Role-Enhanced Zero-Shot Chain-of-Thought Prompting (Software Developer)"
