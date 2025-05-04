import json
import os
import random
from datetime import datetime
from pathlib import Path


import typer

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from transformers import pipeline, set_seed
from typing_extensions import Annotated

from src.config import API_MAX_RETRIES, API_MAX_TIMEOUT, RANDOM_SEED
from src.evaluation_facade import get_model, get_task, get_prompting, get_model_name
from src.models.data_item import DataItem
from src.models.types import LLMs, Tasks, Prompting

app = typer.Typer()


@app.command()
def run_evaluation(
        models: Annotated[
            list[LLMs], typer.Option("--models", "-m", help="Models to perform tasks")
        ],
        tasks: Annotated[
            list[Tasks], typer.Option("--tasks", "-t", help="Tasks to perform")
        ],
        prompting: Annotated[
            list[Prompting],
            typer.Option("--prompting.py", "-p", help="Prompting technique to use"),
        ],
        output_path: Annotated[
            str, typer.Option("--results-path", "-r", help="Path to store results")
        ] = "results",
        evaluate_only: Annotated[
            bool,
            typer.Option(
                "--evaluate-only", "-e", help="Evaluate only, do not run inference"
            ),
        ] = False,
        existing_result_root_path: Annotated[
            str,
            typer.Option(
                "--existing-result-root-path", "-e-r", help="Root path to existing results"
            ),
        ] = "results",
):
    if len(models) > 1 and LLMs.all in models:
        raise ValueError("Cannot specify multiple models and 'all' at the same time")
    if len(tasks) > 1 and Tasks.all in tasks:
        raise ValueError("Cannot specify multiple tasks and 'all' at the same time")
    if len(prompting) > 1 and Prompting.all in prompting:
        raise ValueError(
            "Cannot specify multiple prompting.py techniques and 'all' at the same time"
        )

    if evaluate_only and not Path(existing_result_root_path).exists():
        raise ValueError("Existing result root path does not exist")

    if LLMs.all in models:
        models = [m for m in LLMs if m != LLMs.all]
    if Tasks.all in tasks:
        tasks = [t for t in Tasks if t != Tasks.all]
    if Prompting.all in prompting:
        prompting = [p for p in Prompting if p != Prompting.all]

    logger.info(
        f"Models: {', '.join([m.value for m in models])} | Tasks: {', '.join([t.value for t in tasks])} "
        f"| Prompting: {', '.join([p.value for p in prompting])}"
    )

    if evaluate_only:
        logger.info("Evaluation only mode enabled. Will not run inference.")

    results_path = Path(output_path)
    results_path.mkdir(parents=True, exist_ok=True)

    for model in models:
        model_path = results_path / model.value
        model_path.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            task_path = model_path / task.value
            task_path.mkdir(parents=True, exist_ok=True)
            for pe_technique in prompting:
                pe_path = task_path / pe_technique.value
                pe_path.mkdir(parents=True, exist_ok=True)

    for model in models:
        if not evaluate_only:
            client = None
            if (model is LLMs.phi_4_14b
                or model is LLMs.mistral_nemo_12b
                or model is LLMs.qwen_2_5_14b
                or model is LLMs.granite_3_2_2b
                or model is LLMs.llama_3_2_3b
                or model is LLMs.qwen_2_5_3b
                or model is LLMs.gemma_2_2b
                or model is LLMs.phi_4_mini_3b
                or model is LLMs.granite_3_2_8b
                or model is LLMs.llama_3_1_8b
                or model is LLMs.mistral_7b
                or model is LLMs.qwen_2_5_7b
                or model is LLMs.gemma_2_9b
                or model is LLMs.gemma_3_12b
                or model is LLMs.granite_3_3_2b
                or model is LLMs.granite_3_3_8b
            ):
                client = OpenAI( base_url="http://localhost:11434/v1", api_key="ollama",max_retries=API_MAX_RETRIES, timeout=API_MAX_TIMEOUT)
            elif model is LLMs.gpt_four_1_mini or model is LLMs.gpt_four_1_nano or  model is LLMs.gpt_four_1 or model is LLMs.gpt_four_o or model is LLMs.gpt_four_o_mini:
                client = OpenAI(max_retries=API_MAX_RETRIES, timeout=API_MAX_TIMEOUT)
            llm = get_model(model, client)

        for task in tasks:
            selected_task = get_task(task)
            task_list = selected_task.get_task_list()
            for pe_technique in prompting:
                logger.info(
                    f"Running {model.value} on {task.value} with {pe_technique.value} prompting"
                )
                selected_prompting = get_prompting(pe_technique)
                current_path = (
                        results_path / model.value / task.value / pe_technique.value
                )


                summary_file = current_path / "summary.json"
                if not summary_file.exists():
                    summary_file.touch()
                    summary_file.write_text(
                        json.dumps(
                            {
                                "model": model.value,
                                "task": task.value,
                                "pe_technique": pe_technique.value,
                                "current_index": 0,
                                "is_processing": False,
                                "correct": 0,
                                "total": len(task_list),
                                "accuracy": 0,
                                "created_at": datetime.now().strftime(
                                    "%Y-%m-%d-%H-%M-%S"
                                ),
                            }
                        )
                    )

                summary = json.loads(summary_file.read_text())
                correct = summary["correct"]

                i: int
                item: DataItem
                for i, item in enumerate(task_list):
                    logger.info(f"Running {model.value}")

                    if (current_path / f"{i}.json").exists() and not summary[
                        "is_processing"
                    ]:
                        logger.info(f"Skipping {i}")
                        continue

                    summary["is_processing"] = True
                    summary_file.write_text(json.dumps(summary))
                    prompt = selected_prompting.get_prompt(
                        item.prompt
                    )
                    logger.debug(prompt)
                    if not evaluate_only:
                        response, metadata = llm.inference(
                            prompt, model_name=model.value
                        )
                    else:
                        existing_result_file_path = (
                                Path(existing_result_root_path)
                                / model.value
                                / task.value
                                / pe_technique.value
                                / f"{i}.json"
                        )
                        result_obj = json.loads(existing_result_file_path.read_text())
                        response = result_obj["response"]
                        metadata = {
                            "length": result_obj["length"],
                            "time_taken": result_obj["time_taken"],
                            "start_time": result_obj["start_time"],
                            "end_time": result_obj["end_time"],
                        }
                    is_correct, extracted_answer = selected_task.evaluate(
                        response, item.label
                    )

                    output_path = current_path / f"{i}.json"
                    output_path.touch(exist_ok=True)
                    output_path.write_text(
                        json.dumps(
                            {
                                "prompt": prompt,
                                "response": response,
                                "label": item.label,
                                "is_correct": is_correct,
                                "extracted_answer": extracted_answer,
                                "model": model.value,
                                "task": task.value,
                                "pe_technique": pe_technique.value,
                                "length": metadata["length"],
                                "time_taken": metadata["time_taken"],
                                "start_time": metadata["start_time"],
                                "end_time": metadata["end_time"],
                                "created_at": datetime.now().strftime(
                                    "%Y-%m-%d-%H-%M-%S"
                                ),
                            }
                        )
                    )

                    if is_correct:
                        correct += 1

                    summary = json.loads(summary_file.read_text())
                    summary["current_index"] = i + 1
                    summary["correct"] = correct
                    summary["accuracy"] = correct / len(task_list) * 100
                    summary_file.write_text(json.dumps(summary))

                    logger.info(
                        f"Correct: {correct}/{i + 1} ({correct / (i + 1) * 100:.2f}%)"
                    )

                    summary["is_processing"] = False
                    summary_file.write_text(json.dumps(summary))


if __name__ == "__main__":
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger.add("logs/{time}.log")
    load_dotenv()
    random.seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)


    app()
