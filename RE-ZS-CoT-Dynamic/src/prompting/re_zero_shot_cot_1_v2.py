from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class REZSCOT1V2(Prompting):
    """(Variant 2). Now supports dynamic roles placed after the prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    # MODIFIED: Added role parameter
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None, role: str = None) -> str:
        # MODIFIED: Switched to a template with a placeholder
        template = "\n You are an {role}. Use your knowledge as an {role} to think step-by-step before answering.\n"
        
        final_role = "Expert Mathematician"  # Default role

        if role:
            if "expert" not in role.lower():
                final_role = f"Expert {role}"
            else:
                final_role = role
        
        # MODIFIED: The main prompt comes first, followed by the formatted role instruction
        return prompt + template.format(role=final_role)

    def __str__(self) -> str:
        # MODIFIED: Made the name more generic
        return "Role-Enhanced Zero-Shot Chain-of-Thought Prompting (V2 - Role After)"

