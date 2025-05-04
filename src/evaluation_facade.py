from openai import Client
from transformers import Pipeline

from src.config import (
    gpt_four_1,
    gpt_four_1_mini,
    gpt_four_1_nano,
    gpt_four_o,
    gpt_four_o_mini,
    llama_three_three_instruct_SEVENTY_MODEL,
    phi_4_14b,
    mistral_nemo_12b,
    qwen_2_5_14b,
    granite_3_2_2b,
    llama_3_2_3b,
    qwen_2_5_3b,
    gemma_2_2b,
    phi_4_mini_3b,
    granite_3_2_8b,
    llama_3_1_8b,
    mistral_7b,
    qwen_2_5_7b,
    gemma_2_9b,
    gemma_3_12b,
    granite_3_3_2b,
    granite_3_3_8b,
   
)
from src.llms.gpt import GPT
from src.models.types import LLMs, Tasks, Prompting
#from src.prompting.chain_of_thought import ChainOfThought
#from src.prompting.few_shot import FewShot
#from src.prompting.null_shot import NullShot
from src.prompting.zero_shot import ZeroShot
from src.prompting.zero_shot_chain_of_thought import ZeroShotChainOfThought
from src.prompting.re_zero_shot_cot_1_v1 import REZSCOT1V1
from src.prompting.re_zero_shot_cot_1_v2 import REZSCOT1V2
from src.prompting.re_zero_shot_cot_1_v3 import REZSCOT1V3
from src.prompting.re_zero_shot_cot_2_v1 import REZSCOT2V1
from src.prompting.re_zero_shot_cot_2_v2 import REZSCOT2V2
from src.prompting.re_zero_shot_cot_2_v3 import REZSCOT2V3
from src.prompting.re_zero_shot_cot_3_v1 import REZSCOT3V1
from src.prompting.re_zero_shot_cot_3_v2 import REZSCOT3V2
from src.prompting.re_zero_shot_cot_3_v3 import REZSCOT3V3
from src.prompting.re_zero_shot_cot_4_v1 import REZSCOT4V1
from src.prompting.re_zero_shot_cot_4_v2 import REZSCOT4V2
from src.prompting.re_zero_shot_cot_4_v3 import REZSCOT4V3
from src.prompting.re_zero_shot_cot_5_v1 import REZSCOT5V1
from src.prompting.re_zero_shot_cot_5_v2 import REZSCOT5V2
from src.prompting.re_zero_shot_cot_5_v3 import REZSCOT5V3
from src.prompting.re_zero_shot_cot_6_v1 import REZSCOT6V1
from src.prompting.re_zero_shot_cot_6_v2 import REZSCOT6V2
from src.prompting.re_zero_shot_cot_6_v3 import REZSCOT6V3
from src.prompting.re_zero_shot_cot_7_v1 import REZSCOT7V1
from src.prompting.re_zero_shot_cot_7_v2 import REZSCOT7V2
from src.prompting.re_zero_shot_cot_7_v3 import REZSCOT7V3
from src.prompting.re_zero_shot_cot_8_v1 import REZSCOT8V1
from src.prompting.re_zero_shot_cot_8_v2 import REZSCOT8V2
from src.prompting.re_zero_shot_cot_8_v3 import REZSCOT8V3

#from src.prompting.re_zero_shot_cot_v9 import REZSCOTV9
#from src.prompting.re_zero_shot_cot_v10 import REZSCOTV10
#from src.tasks.anli import ANLI
from src.tasks.aqua import AQuA
from src.tasks.commonsense_qa import CommonsenseQA
from src.tasks.gsm8k import GSM8K
from src.tasks.math_algebra import MATHAlgebra
from src.tasks.math_counting_and_probability import MATHCountingAndProbability
from src.tasks.math_geometry import MATHGeometry
from src.tasks.math_intermediate_algebra import MATHIntermediateAlgebra
from src.tasks.math_number_theory import MATHNumberTheory
from src.tasks.math_prealgebra import MATHPreAlgebra
from src.tasks.math_precalculus import MATHPreCalculus
from src.tasks.open_book_qa import OpenBookQA
from src.tasks.race_h import RACEHigh
from src.tasks.race_m import RACEMiddle
from src.tasks.strategyqa import StrategyQA
#from src.tasks.svamp import SVAMP
from src.tasks.triviaqa import TriviaQA
from src.tasks.winogrande import Winogrande
from src.tasks.wmt_news_en_ja import WMTENJA
from src.tasks.wmt_news_ja_en import WMTJAEN


def get_prompting(prompting: Prompting):
    match prompting:
        case Prompting.zero_shot:
            return ZeroShot
        #case Prompting.few_shot:
            #return FewShot
        #case Prompting.null_shot:
         #   return NullShot
        #case Prompting.chain_of_thought:
            #return ChainOfThought
        case Prompting.zero_shot_chain_of_thought:
            return ZeroShotChainOfThought
        case Prompting.re_zero_shot_cot_1_v1:
            return REZSCOT1V1
        case Prompting.re_zero_shot_cot_1_v2:
            return REZSCOT1V2
        case Prompting.re_zero_shot_cot_1_v3:
            return REZSCOT1V3
        case Prompting.re_zero_shot_cot_2_v1:
            return REZSCOT2V1
        case Prompting.re_zero_shot_cot_2_v2:
            return REZSCOT2V2
        case Prompting.re_zero_shot_cot_2_v3:
            return REZSCOT2V3
        case Prompting.re_zero_shot_cot_3_v1:
            return REZSCOT3V1
        case Prompting.re_zero_shot_cot_3_v2:
            return REZSCOT3V2
        case Prompting.re_zero_shot_cot_3_v3:
            return REZSCOT3V3
        case Prompting.re_zero_shot_cot_4_v1:
            return REZSCOT4V1
        case Prompting.re_zero_shot_cot_4_v2:
            return REZSCOT4V2
        case Prompting.re_zero_shot_cot_4_v3:
            return REZSCOT4V3
        case Prompting.re_zero_shot_cot_5_v1:
            return REZSCOT5V1
        case Prompting.re_zero_shot_cot_5_v2:
            return REZSCOT5V2
        case Prompting.re_zero_shot_cot_5_v3:
            return REZSCOT5V3
        case Prompting.re_zero_shot_cot_6_v1:
            return REZSCOT6V1
        case Prompting.re_zero_shot_cot_6_v2:
            return REZSCOT6V2
        case Prompting.re_zero_shot_cot_6_v3:
            return REZSCOT6V3
        case Prompting.re_zero_shot_cot_7_v1:
            return REZSCOT7V1
        case Prompting.re_zero_shot_cot_7_v2:
            return REZSCOT7V2
        case Prompting.re_zero_shot_cot_7_v3:
            return REZSCOT7V3
        case Prompting.re_zero_shot_cot_8_v1:
            return REZSCOT8V1
        case Prompting.re_zero_shot_cot_8_v2:
            return REZSCOT8V2
        case Prompting.re_zero_shot_cot_8_v3:
            return REZSCOT8V3

        #case Prompting.re_zero_shot_cot_v9:
         #   return REZSCOTV9
       # case Prompting.re_zero_shot_cot_v10:
        #    return REZSCOTV10

        case _:
            raise NotImplementedError(f"Prompting {prompting.value} not implemented")


def get_model(model: LLMs, client: Client | Pipeline):
    match model:
        case LLMs.gpt_four_1_mini | LLMs.gpt_four_1_nano | LLMs.gpt_four_1 | LLMs.gpt_four_o | LLMs.gpt_four_o_mini | LLMs.llama_three_three_instruct_SEVENTY_MODEL | LLMs.phi_4_14b | LLMs.mistral_nemo_12b | LLMs.qwen_2_5_14b | LLMs.granite_3_2_2b | LLMs.llama_3_2_3b | LLMs.qwen_2_5_3b | LLMs.gemma_2_2b | LLMs.phi_4_mini_3b | LLMs.granite_3_2_8b | LLMs.llama_3_1_8b | LLMs.mistral_7b | LLMs.qwen_2_5_7b | LLMs.gemma_2_9b | LLMs.granite_3_3_8b | LLMs.granite_3_3_2b | LLMs.gemma_3_12b:
            return GPT(client)
        case _:
            raise NotImplementedError(f"Model {model.value} not implemented")


def get_model_name(model: LLMs):
    match model:
        case LLMs.gpt_four_1_mini:
            return gpt_four_1_mini
        case LLMs.gpt_four_1_nano:
            return gpt_four_1_nano
        case LLMs.gpt_four_1:
            return gpt_four_1
        case LLMs.gpt_four_o:
            return gpt_four_o
        case LLMs.gpt_four_o_mini:
            return gpt_four_o_mini
        case LLMs.llama_three_three_instruct_SEVENTY_MODEL:
            return llama_three_three_instruct_SEVENTY_MODEL
        case LLMs.phi_4_14b:
            return phi_4_14b
        case LLMs.mistral_nemo_12b:
            return mistral_nemo_12b
        case LLMs.qwen_2_5_14b:
            return qwen_2_5_14b
        case LLMs.granite_3_2_2b:
            return granite_3_2_2b
        case LLMs.granite_3_3_2b:
            return granite_3_3_2b
        case LLMs.llama_3_2_3b:
            return llama_3_2_3b
        case LLMs.qwen_2_5_3b:
            return qwen_2_5_3b
        case LLMs.gemma_2_2b:
            return gemma_2_2b
        case LLMs.phi_4_mini_3b:
            return phi_4_mini_3b
        case LLMs.granite_3_2_8b:
            return granite_3_2_8b
        case LLMs.granite_3_3_8b:
            return granite_3_3_8b
        case LLMs.llama_3_1_8b:
            return llama_3_1_8b
        case LLMs.mistral_7b:
            return mistral_7b
        case LLMs.qwen_2_5_7b:
            return qwen_2_5_7b
        case LLMs.gemma_2_9b:
            return gemma_2_9b
        case LLMs.gemma_3_12b:
            return gemma_3_12b
        case _:
            raise NotImplementedError(f"Model {model.value} not implemented")


def get_task(task: Tasks):
    match task:
        case Tasks.TriviaQA:
            return TriviaQA
        case Tasks.RACE_H:
            return RACEHigh
        case Tasks.RACE_M:
            return RACEMiddle
        case Tasks.Winogrande:
            return Winogrande
        case Tasks.CommonsenseQA:
            return CommonsenseQA
        case Tasks.StrategyQA:
            return StrategyQA
        case Tasks.OpenBookQA:
            return OpenBookQA
        case Tasks.AQuA:
            return AQuA
        case Tasks.GSM8K:
            return GSM8K
        #case Tasks.SVAMP:
          #  return SVAMP
        case Tasks.MATHAlgebra:
            return MATHAlgebra
        case Tasks.MATHCountingAndProbability:
            return MATHCountingAndProbability
        case Tasks.MATHGeometry:
            return MATHGeometry
        case Tasks.MATHNumberTheory:
            return MATHNumberTheory
        case Tasks.MATHIntermediateAlgebra:
            return MATHIntermediateAlgebra
        case Tasks.MATHPreAlgebra:
            return MATHPreAlgebra
        case Tasks.MATHPreCalculus:
            return MATHPreCalculus
        case _:
            raise NotImplementedError(f"Task {task.value} not implemented")
