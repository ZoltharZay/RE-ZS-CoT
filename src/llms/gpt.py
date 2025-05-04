from time import perf_counter, sleep

import openai
from loguru import logger

from src.config import RANDOM_SEED,gemma_3_12b, gpt_four_1,gpt_four_1_mini,gpt_four_1_nano,gpt_four_o, gpt_four_o_mini,llama_three_three_instruct_SEVENTY_MODEL, phi_4_14b,mistral_nemo_12b,qwen_2_5_14b,granite_3_2_2b, llama_3_2_3b, granite_3_2_2b, qwen_2_5_3b, gemma_2_2b,phi_4_mini_3b,granite_3_2_8b,llama_3_1_8b, mistral_7b, qwen_2_5_7b,gemma_2_9b, granite_3_3_8b,granite_3_3_2b        
from src.llms.llm import LLM


class GPT(LLM):
    """GPT"""

    def __init__(self, client: openai.Client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

    def inference(self, prompt: str, model_name="") -> (str, dict):
        model = ""
        match model_name:
            case "gpt-4.1-2025-04-14":
                model = gpt_four_1
            case "gpt-4.1-mini-2025-04-14":
                model = gpt_four_1_mini
            case "gpt-4.1-nano-2025-04-14":
                model = gpt_four_1_nano
            case "gpt-4o-2024-08-06":
                model = gpt_four_o
            case "gpt-4o-mini-2024-07-18":
                model = gpt_four_o_mini
            case "llama-3.3-70b-instruct":
                model =  llama_three_three_instruct_SEVENTY_MODEL
            case "phi4:14b-q4_K_M":
                model = phi_4_14b
            case "mistral-nemo:12b-instruct-2407-q4_K_M":
                model = mistral_nemo_12b
            case "qwen2.5:14b-instruct-q4_K_M":
                model = qwen_2_5_14b
            case "granite3.2:2b-instruct-q4_K_M":
                model = granite_3_2_2b
            case "granite3.3:2b":
                model = granite_3_3_2b
            case "llama3.2:3b-instruct-q4_K_M":
                model = llama_3_2_3b
            case "qwen2.5:3b-instruct-q4_K_M":
                model = qwen_2_5_3b
            case "gemma2:2b-instruct-q4_K_M":
                model = gemma_2_2b
            case "phi4-mini:3.8b-q4_K_M":
                model = phi_4_mini_3b
            case "granite3.2:8b-instruct-q4_K_M":
                model = granite_3_2_8b
            case "granite3.3:8b":
                model = granite_3_3_8b
            case "llama3.1:8b-instruct-q4_K_M":
                model = llama_3_1_8b
            case "mistral:7b-instruct-q4_K_M":
                model = mistral_7b
            case "qwen2.5:7b-instruct-q4_K_M":
                model = qwen_2_5_7b
            case "gemma2:9b-instruct-q4_K_M":
                model = gemma_2_9b
            case "gemma3:12b-it-q4_K_M":
                model = gemma_3_12b
            case _:
                raise NotImplementedError(f"Model {model_name} not implemented")
        logger.info(f"Generating response from {model_name}")
        start_time = perf_counter()
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt},{"role": "user", "content": prompt}],
                temperature=0,
                seed=RANDOM_SEED,
            )
            end_time = perf_counter()
            response = chat_completion.choices[0].message.content
        except openai.RateLimitError as e:
            print(e)
            raise e
        except openai.OpenAIError as e:
            end_time = perf_counter()
            response = f"ERROR: {e}"
        logger.debug(response)
        logger.success(
            f"Response generated from {model}, response length: {len(response)}, "
            f"time taken: {end_time - start_time} seconds")
        return response, {"length": len(response), "time_taken": end_time - start_time, "start_time": start_time,
                          "end_time": end_time}

    def __str__(self) -> str:
        return "GPT"
