from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class REZSCOT2V3(Prompting):
    """(Variant 3). Now supports dynamic roles split around the prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    # MODIFIED: Added role parameter
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None, role: str = None) -> str:
        # MODIFIED: Split the original instruction into two templates
        template1 = "You are an {role}.\n"
        template2 = "\n Use your knowledge as an {role} to answer the question."
        
        final_role = "Expert Mathematician"  # Default role

        if role:
            if "expert" not in role.lower():
                final_role = f"Expert {role}"
            else:
                final_role = role
        
        # MODIFIED: The prompt is placed between the two formatted role instructions
        return template1.format(role=final_role) + prompt + template2.format(role=final_role)

    def __str__(self) -> str:
        # MODIFIED: Made the name more generic
        return "Role-Enhanced Zero-Shot Prompting (V3 - Role Split)"

