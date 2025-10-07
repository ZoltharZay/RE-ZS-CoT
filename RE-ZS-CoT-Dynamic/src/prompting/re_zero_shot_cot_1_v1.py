from src.models.data_item import DataItem
from src.prompting.prompting import Prompting


class REZSCOT1V1(Prompting):
    """(Variant 1). Now supports dynamic roles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_prompt(cls, prompt: str, examples: list[DataItem] = None, role: str = None) -> str:
        # This template contains the {role} placeholder where the dynamic role will be inserted.
        template = "You are an {role}. Use your knowledge as an {role} to think step-by-step before answering.\n"
        
        final_role = "Expert Mathematician"  # Default role if none is provided.

        if role:
            # Check if the provided role string already contains "expert" (case-insensitive)
            if "expert" not in role.lower():
                # If not, prepend "Expert " to the role
                final_role = f"Expert {role}"
            else:
                # Otherwise, use the role as is
                final_role = role
        
        # Format the template with the final role and prepend it to the user's prompt
        return template.format(role=final_role) + prompt

    def __str__(self) -> str:
        # Made the name more generic as it's no longer limited to Mathematician
        return "Role-Enhanced Zero-Shot Chain-of-Thought Prompting"