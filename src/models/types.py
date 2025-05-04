from enum import Enum


class LLMs(Enum):
    gpt_four_1 = "gpt-4.1-2025-04-14"
    gpt_four_1_mini = "gpt-4.1-mini-2025-04-14"
    gpt_four_1_nano = "gpt-4.1-nano-2025-04-14"
    gpt_four_o = "gpt-4o-2024-08-06"
    gpt_four_o_mini = "gpt-4o-mini-2024-07-18"

    #70b model
    llama_three_three_instruct_SEVENTY_MODEL= "llama-3.3-70b-instruct"

    #12-14b
    phi_4_14b = "phi4:14b-q4_K_M"
    mistral_nemo_12b = "mistral-nemo:12b-instruct-2407-q4_K_M"
    qwen_2_5_14b= "qwen2.5:14b-instruct-q4_K_M"
    gemma_3_12b = "gemma3:12b-it-q4_K_M"
    


    #2-3b
    granite_3_2_2b = "granite3.2:2b-instruct-q4_K_M"
    granite_3_3_2b = "granite3.3:2b"
    llama_3_2_3b = "llama3.2:3b-instruct-q4_K_M"
    qwen_2_5_3b = "qwen2.5:3b-instruct-q4_K_M"
    gemma_2_2b = "gemma2:2b-instruct-q4_K_M"
    phi_4_mini_3b = "phi4-mini:3.8b-q4_K_M"

    #7-9b
    granite_3_2_8b = "granite3.2:8b-instruct-q4_K_M"
    granite_3_3_8b = "granite3.3:8b"
    llama_3_1_8b = "llama3.1:8b-instruct-q4_K_M"
    mistral_7b = "mistral:7b-instruct-q4_K_M"
    qwen_2_5_7b = "qwen2.5:7b-instruct-q4_K_M"
    gemma_2_9b = "gemma2:9b-instruct-q4_K_M"





    all = "all"


class Tasks(Enum):
    all = "all"
    # Closed-book QA
    TriviaQA = "triviaqa"
    # Reading comprehension
    RACE_H = "race-h"
    RACE_M = "race-m"
    # Winogrande
    Winogrande = "winogrande"
    # Commonsense reasoning
    CommonsenseQA = "csqa"
    StrategyQA = "strategyqa"
    OpenBookQA = "openbookqa"
    # Arithmetic reasoning
    AQuA = "aqua"
    GSM8K = "gsm8k"
    #SVAMP = "svamp"
    # MATH
    MATHAlgebra = "math-algebra"
    MATHCountingAndProbability = "math-count-prob"
    MATHGeometry = "math-geometry"
    MATHNumberTheory = "math-number"
    MATHIntermediateAlgebra = "math-int-algebra"
    MATHPreAlgebra = "math-pre-algebra"
    MATHPreCalculus = "math-pre-calc"


class Prompting(Enum):
    zero_shot = "zero-shot"
    #few_shot = "few-shot"
    #null_shot = "null-shot"
    #chain_of_thought = "cot"
    zero_shot_chain_of_thought = "zero-shot-cot"
    re_zero_shot_cot_1_v1 = "re-1-1"
    re_zero_shot_cot_1_v2 = "re-1-2"
    re_zero_shot_cot_1_v3 = "re-1-3"
    re_zero_shot_cot_2_v1 = "re-2-1"
    re_zero_shot_cot_2_v2 = "re-2-2"
    re_zero_shot_cot_2_v3 = "re-2-3"
    re_zero_shot_cot_3_v1 = "re-3-1"
    re_zero_shot_cot_3_v2 = "re-3-2"
    re_zero_shot_cot_3_v3 = "re-3-3"
    re_zero_shot_cot_4_v1 = "re-4-1"
    re_zero_shot_cot_4_v2 = "re-4-2"
    re_zero_shot_cot_4_v3 = "re-4-3"
    re_zero_shot_cot_5_v1 = "re-5-1"
    re_zero_shot_cot_5_v2 = "re-5-2"
    re_zero_shot_cot_5_v3 = "re-5-3"
    re_zero_shot_cot_6_v1 = "re-6-1"
    re_zero_shot_cot_6_v2 = "re-6-2"
    re_zero_shot_cot_6_v3 = "re-6-3"
    re_zero_shot_cot_7_v1 = "re-7-1"
    re_zero_shot_cot_7_v2 = "re-7-2"
    re_zero_shot_cot_7_v3 = "re-7-3"
    re_zero_shot_cot_8_v1 = "re-8-1"
    re_zero_shot_cot_8_v2 = "re-8-2"
    re_zero_shot_cot_8_v3 = "re-8-3"
    #re_zero_shot_cot_v9 = "re_zero_shot_cot_v9"
    #re_zero_shot_cot_v10 = "re_zero_shot_cot_v10"


    all = "all"
