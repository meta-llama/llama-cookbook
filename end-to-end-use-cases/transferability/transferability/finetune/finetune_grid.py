import os
import subprocess
from pathlib import Path

from ..utils import load_config


def get_general_finetune_args(finetuning_config, output_dir):
    experiment_dir = Path(output_dir).parent
    model_path = finetuning_config["model_path"]
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model path {model_path} does not exist")
    tokenizer_path = finetuning_config["tokenizer_path"]

    # TODO: Change "task1" to task name defined in config
    dataset_path = (
        experiment_dir / "formatted_datasets" / "task1" / "train_conversation_data.json"
    )

    return [
        f"dataset.dataset_path={dataset_path}",
        f"checkpointer.checkpoint_dir={model_path}",
        f"tokenizer.path={tokenizer_path}",
        f"epochs={finetuning_config['epochs']}",
        f"batch_size={finetuning_config['batch_size']}",
        f"metric_logger.log_dir={experiment_dir}/finetune_logs",
    ]


def build_fft_jobs(config, output_dir):
    """Build FFT (Full Fine-Tuning) jobs based on config"""
    jobs = []
    finetuning_config = config["finetuning"]

    recipe = (
        "full_finetune_distributed"
        if finetuning_config.get("distributed")
        else "full_finetune_single_device"
    )

    torchtune_config = finetuning_config.get("fft_torchtune_config")
    base_cmd = [
        "tune",
        "run",
        "--nproc_per_node",
        str(finetuning_config["ngpu"]),
        recipe,
        "--config",
        torchtune_config,
    ]

    base_cmd += get_general_finetune_args(finetuning_config, output_dir)

    # Build list of modules to train based on config
    modules_to_train = []
    if finetuning_config.get("fusion", False):
        modules_to_train.append("fusion")
    if finetuning_config.get("fusion+encoder", False):
        modules_to_train.append("fusion+encoder")
    if finetuning_config.get("fusion+decoder", False):
        modules_to_train.append("fusion+decoder")
    if finetuning_config.get("fusion+encoder+decoder", False):
        modules_to_train.append("fusion+encoder+decoder")

    for modules in modules_to_train:
        op_path = f"{output_dir}/full_{modules}"
        if os.path.exists(op_path):
            print(f"Skipping {op_path} as it already exists")
            continue
        module_opts = [f"model.{mod}_trainable=True" for mod in modules.split("+")]
        jobs.append(base_cmd + [f"output_dir={op_path}"] + module_opts)

    return jobs


def build_lora_jobs(config, output_dir):
    """Build LoRA jobs based on config"""
    jobs = []
    finetuning_config = config["finetuning"]

    if not finetuning_config.get("lora_ranks"):
        return jobs

    recipe = (
        "lora_finetune_distributed"
        if finetuning_config.get("distributed")
        else "lora_finetune_single_device"
    )

    torchtune_config = finetuning_config.get("lora_torchtune_config")

    base_cmd = [
        "tune",
        "run",
        "--nproc_per_node",
        str(finetuning_config["ngpu"]),
        recipe,
        "--config",
        torchtune_config,
    ]

    base_cmd += get_general_finetune_args(finetuning_config, output_dir)

    for rank in finetuning_config["lora_ranks"]:
        op_path = f"{output_dir}/lora_{rank}"
        if os.path.exists(op_path):
            print(f"Skipping {op_path} as it already exists")
            continue
        jobs.append(
            base_cmd
            + [
                f"output_dir={op_path}",
                f"model.lora_rank={rank}",
                f"model.lora_alpha={int(rank)*2}",
            ]
        )

    return jobs


def run_finetune_grid(experiment_dir: str):
    print("🚀 Starting fine-tuning grid execution...")
    print(f"📁 Experiment directory: {experiment_dir}")

    # Get script directory and config path
    script_dir = Path(__file__).parent.parent.parent
    config_path = script_dir / "config.yaml"
    print(f"📝 Loading configuration from: {config_path}")

    # Load configuration
    config = load_config(config_path)
    print("✅ Configuration loaded successfully")

    # Set output directory
    output_dir = Path(experiment_dir) / "finetuned_checkpoints"
    print(f"💾 Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print("📂 Output directory created/verified")

    # Build all jobs
    all_jobs = []
    print("\n🔧 Building fine-tuning jobs...")

    # Check if we should run FFT jobs (if any fusion settings are enabled)
    finetuning_config = config["finetuning"]
    if any(
        [
            finetuning_config.get("fusion", False),
            finetuning_config.get("fusion+encoder", False),
            finetuning_config.get("fusion+decoder", False),
            finetuning_config.get("fusion+encoder+decoder", False),
        ]
    ):
        print("🔄 Building Full Fine-Tuning (FFT) jobs...")
        fft_jobs = build_fft_jobs(config, output_dir)
        all_jobs.extend(fft_jobs)
        print(f"✅ Built {len(fft_jobs)} FFT jobs")

        # Print details of FFT jobs
        for i, job in enumerate(fft_jobs, 1):
            job_type = "FFT"
            modules = [arg for arg in job if "trainable=True" in str(arg)]
            if modules:
                module_info = ", ".join(
                    [
                        mod.split(".")[1].replace("_trainable=True", "")
                        for mod in modules
                    ]
                )
                print(f"   📋 FFT Job {i}: {module_info}")

    # Check if we should run LoRA jobs
    if finetuning_config.get("lora_ranks"):
        print("🔄 Building LoRA fine-tuning jobs...")
        lora_jobs = build_lora_jobs(config, output_dir)
        all_jobs.extend(lora_jobs)
        lora_count = len(lora_jobs)
        print(f"✅ Built {lora_count} LoRA jobs")

        # Print details of LoRA jobs
        ranks = finetuning_config.get("lora_ranks", [])
        for i, rank in enumerate(ranks, 1):
            print(f"   📋 LoRA Job {i}: rank={rank}, alpha={rank*2}")

    total_jobs = len(all_jobs)
    print(f"\n📊 Total jobs to execute: {total_jobs}")

    # Run all jobs
    print(f"\n🎯 Executing {total_jobs} fine-tuning jobs...")
    print("=" * 60)

    for job_idx, job in enumerate(all_jobs, 1):
        print(f"\n📈 Job {job_idx}/{total_jobs} - Starting...")

        # Extract job type and details for better logging
        job_type = "LoRA" if "lora_finetune" in " ".join(job) else "FFT"
        output_path = next(
            (arg.split("=")[1] for arg in job if arg.startswith("output_dir=")),
            "unknown",
        )
        job_name = Path(output_path).name if output_path != "unknown" else "unknown"

        print(f"🔧 Type: {job_type}")
        print(f"📁 Output: {job_name}")
        # print(f"⚡ Command: {' '.join(map(str, job))}")
        print("-" * 40)

        try:
            print(f"⏳ Executing job {job_idx}/{total_jobs}...")
            subprocess.run(job, check=True, capture_output=False)
            print(f"✅ Job {job_idx}/{total_jobs} completed successfully!")

        except subprocess.CalledProcessError as e:
            print(
                f"❌ Job {job_idx}/{total_jobs} failed with return code {e.returncode}"
            )
            print(f"💥 Error: {e}")
            raise
        except Exception as e:
            print(f"❌ Job {job_idx}/{total_jobs} failed with unexpected error: {e}")
            raise

    print("\n" + "=" * 60)
    print("🎉 All fine-tuning jobs completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
    print("🏁 Fine-tuning grid execution finished.")


if __name__ == "__main__":
    run_finetune_grid("experiments/w2_ocr")
