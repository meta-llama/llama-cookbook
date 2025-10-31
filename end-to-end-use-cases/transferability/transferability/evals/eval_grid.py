import json
from pathlib import Path
from typing import Optional

from ..utils import load_config

from .grader import get_grader

from .inference import create_inference_request, LocalModelRunner

from .shift_analysis import calculate_transferability_index


def load_dataset(dataset_path: Path, nb_samples: Optional[int] = None):
    """Load conversation dataset from JSON file."""
    with open(dataset_path, "r") as f:
        samples = json.load(f)
    if nb_samples is not None:
        samples = samples[:nb_samples]
    return samples


def grade_dataset(llm_runner, dataset, grader, inference_params):
    """Grade a dataset using the LLM runner."""
    requests = [create_inference_request(m, **inference_params) for m in dataset]
    llm_outputs = llm_runner.run_batch(requests)
    rows = [
        {"expected_output": m[-1]["content"][0]["text"], "raw_response": l}
        for m, l in zip(dataset, llm_outputs)
    ]
    return grader.grade(rows)


def get_finetuned_checkpoint_path(base_path: Path, checkpoint_to_eval: int):
    """Get the path to a specific finetuned checkpoint."""
    if checkpoint_to_eval == -1:
        # Find the last checkpoint
        checkpoint_dirs = []
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir() and item.name.startswith("epoch_"):
                    try:
                        epoch_num = int(item.name.split("_")[1])
                        checkpoint_dirs.append((epoch_num, item))
                    except (ValueError, IndexError):
                        continue

        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoints found in {base_path}")

        # Return the highest epoch
        return max(checkpoint_dirs, key=lambda x: x[0])[1]
    else:
        # Return specific epoch
        epoch_path = base_path / f"epoch_{checkpoint_to_eval}"
        if not epoch_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {epoch_path}")
        return epoch_path


def run_eval_grid(experiment_dir: str):
    print("🚀 Starting evaluation grid execution...")
    print(f"📁 Experiment directory: {experiment_dir}")

    # Get script directory and config path
    script_dir = Path(__file__).parent.parent.parent
    config_path = script_dir / "config.yaml"
    print(f"📝 Loading configuration from: {config_path}")

    logs_dir = Path(experiment_dir) / "grader_logs"

    # Load configuration
    config = load_config(config_path)
    print("✅ Configuration loaded successfully")

    # Load task names
    tasks = ["task1", "task2"]

    # Populate checkpoints dictionary with base and finetuned checkpoints
    print("🔍 Building checkpoint list...")
    checkpoints = {}

    # Add base model from config
    base_model_path = config["finetuning"]["model_path"]
    checkpoints["base_model"] = base_model_path
    print(f"   📋 Base model: {base_model_path}")

    # Add finetuned checkpoints
    finetuned_ckpts_dir = Path(experiment_dir) / "finetuned_checkpoints"
    checkpoint_to_eval = config["evals"]["checkpoint_to_eval"]

    if finetuned_ckpts_dir.exists():
        for ckpt_dir in finetuned_ckpts_dir.iterdir():
            if ckpt_dir.is_dir():
                try:
                    ckpt_path = get_finetuned_checkpoint_path(
                        ckpt_dir, checkpoint_to_eval
                    )
                    model_name = f"finetuned_{ckpt_dir.name}"
                    checkpoints[model_name] = str(ckpt_path)
                    print(f"   📋 Finetuned: {model_name} -> {ckpt_path}")
                except FileNotFoundError as e:
                    print(f"   ⚠️  Skipping {ckpt_dir.name}: {e}")
    else:
        print("   ⚠️  No finetuned checkpoints directory found")

    print(f"📊 Total checkpoints to evaluate: {len(checkpoints)}")

    # Load model server args from config
    model_server_args = config["evals"]["model_server_args"]
    print(f"🔧 Model server args: {model_server_args}")

    # Load inference params from config
    inference_params = config["evals"]["inference_params"]
    print(f"⚡ Inference params: {inference_params}")

    eval_grid_results = []

    print(f"\n🎯 Starting evaluation grid...")
    print("=" * 60)

    total_evaluations = len(checkpoints) * len(tasks)
    eval_count = 0

    for model_name, ckpt in checkpoints.items():
        print(f"\n🤖 Evaluating model: {model_name}")
        print(f"📁 Checkpoint: {ckpt}")

        # Initialize model runner for this checkpoint
        llm_runner = LocalModelRunner(ckpt, **model_server_args)

        # Create log file for this model in `logs_dir`
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = logs_dir / f"{model_name}_evaluation_log.json"
        model_log_data = {
            "model_name": model_name,
            "checkpoint_path": str(ckpt),
            "model_server_args": model_server_args,
            "inference_params": inference_params,
            "tasks": {},
        }
        for task_name in tasks:
            eval_count += 1
            print(f"\n📈 Evaluation {eval_count}/{total_evaluations}")
            print(f"🎯 Model: {model_name}, Task: {task_name}")

            # Get task-specific grader from config
            grader_name = config[task_name].get("grader", "JSONGrader")
            grader = get_grader(grader_name)
            print(f"   🔧 Using grader: {grader_name}")

            dataset_path = (
                Path(experiment_dir)
                / "formatted_datasets"
                / task_name
                / "test_conversation_data.json"
            )

            if not dataset_path.exists():
                print(f"   ❌ Dataset not found: {dataset_path}")
                continue

            print(f"   📊 Loading dataset: {dataset_path}")
            dataset = load_dataset(dataset_path)
            print(f"   📋 Dataset size: {len(dataset)} samples")

            try:
                print("   ⏳ Running evaluation...")
                eval_result = grade_dataset(
                    llm_runner, dataset, grader, inference_params
                )

                # Log eval_result for each task in the log file
                model_log_data["tasks"][task_name] = {
                    "metrics": eval_result.metrics,
                    "topline_metric_name": eval_result.topline_metric_name,
                    "num_samples": len(eval_result.result_data),
                    "result_data": eval_result.result_data,
                    "rows": eval_result.rows,
                }

                topline_metric = eval_result.topline_metric_name
                score = eval_result.metrics.get(topline_metric)

                print(f"   ✅ {topline_metric}: {score:.4f}")

                eval_grid_results.append(
                    {
                        "model": model_name,
                        "task": task_name,
                        "topline_metric": topline_metric,
                        "score": score,
                        "metrics": eval_result.metrics,
                    }
                )

            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                eval_grid_results.append(
                    {
                        "model": model_name,
                        "task": task_name,
                        "topline_metric": "error",
                        "score": -1,
                        "error": str(e),
                    }
                )

        # Write the log file for this model
        with open(log_file_path, "w") as f:
            json.dump(model_log_data, f, indent=2)
        print(f"   📄 Evaluation log saved to: {log_file_path}")

        llm_runner.shutdown()

    # Save results
    results_path = Path(experiment_dir) / "eval_grid_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_grid_results, f, indent=2)

    print("\n" + "=" * 60)
    print("🎉 Evaluation grid completed!")
    print(f"📁 Results saved to: {results_path}")

    # Print summary table
    # print("\n📊 Results Summary:")
    # print("-" * 80)
    # print(f"{'Model':<25} {'Task':<10} {'Metric':<15} {'Score':<10}")
    # print("-" * 80)
    # for result in eval_grid_results:
    #     print(
    #         f"{result['model']:<25} {result['task']:<10} {result['topline_metric']:<15} {result['score']:<10.4f}"
    #     )
    # print("-" * 80)

    transferability_results = calculate_transferability_index(eval_grid_results)

    # Print summary table
    print("\n📊 Results Summary:")
    print("-" * 80)
    print(transferability_results)
    transferability_results_path = Path(experiment_dir) / "transferability_results.csv"
    transferability_results.to_csv(transferability_results_path, index=False)

    return eval_grid_results


if __name__ == "__main__":
    run_eval_grid(
        "/data/users/subramen/fbsource/fbcode/users/subramen/internal-llama-cookbook/end-to-end-use-cases/transferability/experiments/test01"
    )
