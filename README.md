# RE-evaluating LLMs: Role-Enhanced Zero-Shot Chain-of-Thought Prompting for Better Reasoning

This repository contains the code for the paper "RE-evaluating LLMs: Role-Enhanced Zero-Shot Chain-of-Thought Prompting for Better Reasoning".
For data and analysis code, please refer to the [data repository](Link to me available soon).

## Authors
Anonymous



## File structure
```

├── datasets # Supported evaluation datasets
├── main.py # The main entry point of the program, new dataset and model should be added here first
├── requirements.txt # Required Python packages
└── src # Source code
    ├── __init__.py # Initialization file
    ├── config.py # Configuration file, new dataset and model should be added here first
    ├── evaluation_facade.py # Facade for evaluation, new dataset and model should be added here first
    ├── llms # Wrapper for LLMs
    ├── models # OOP models
    ├── prompting # Wrapper for prompt engineering approaches
    └── tasks # Wrapper for evaluation   tasks
```

## Installation and Usage
0. Create a virtual environment (if needed):
```bash
conda create -n re-zs-cot python=3.11
```
and activate it:
```bash
conda activate re-zs-cot
```
1. Make a new `.txt` file and rename it to `.env`. Follow instructions respective model providers to get the necessary API keys.
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. Make sure Ollama is installed and the requried models are downloaded.
 ```bash
ollama run [model name]
```
5. Main program is `main.py`. It can be run with the following command:
```bash
python main.py <command> <options>
```
The main program supports the following commands:
- `--help`, `-h`: Show help message
- `--models`, `-m`: [Required] Models to perform tasks `[gpt-4.1-2025-04-14|gpt-4.1-mini-2025-04-14|gpt-4.1-nano-2025-04-14|gpt-4o-2024-08-06|gpt-4o-mini-2024-07-18|phi4:14b-q4_K_M|mistral-nemo:12b-instruct-2407-q4_K_M|qwen2.5:14b-instruct-q4_K_M|gemma3:12b-it-q4_K_M|granite3.2:2b-instruct-q4_K_M|granite3.3:2b|llama3.2:3b-instruct-q4_K_M|qwen2.5:3b-instruct-q4_K_M|gemma2:2b-instruct-q4_K_M|phi4-mini:3.8b-q4_K_M|granite3.2:8b-instruct-q4_K_M|granite3.3:8b|llama3.1:8b-instruct-q4_K_M|mistral:7b-instruct-q4_K_M|qwen2.5:7b-instruct-q4_K_M|gemma2:9b-instruct-q4_K_M|all]`
- `--tasks`, `-t`: [Required] Tasks to perform `[all|triviaqa|race-h|race-m|winogrande|csqa|strategyqa|openbookqa|aqua|gsm8k|math-algebra|math-count-prob|math-geometry|math-number|math-int-algebra|math-pre-algebra|math-pre-calc]`
- `--prompting`, `-p`: [Required] Prompting technique to use `[zero-shot|few-shot|re-1-1|re-1-2|re-1-3|re-2-1|re-2-2|re-2-3|re-3-1|re-3-2|re-3-3|re-4-1|re-4-2|re-4-3|re-5-1|re-5-2|re-5-3|re-6-1|re-6-2|re-6-3|re-7-1|re-7-2|re-7-3|re-8-1|re-8-2|re-8-3|all]`
- `--output-path`, `-r`: Path to store results `[default: results]`
- `--evaluate-only`, `-e`: Evaluate only, do not run inference
- `--existing-result-root-path`, `-e-r`: Root path to existing results `[default:
                                  results]`

## Example
```bash
python main.py -m gpt-4.1-2025-04-14 -t aqua -p zero-shot
```
