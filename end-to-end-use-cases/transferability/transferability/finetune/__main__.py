"""Entry point for running fine-tuning grid as a module."""

if __name__ == "__main__":
    import sys

    from .finetune_grid import run_finetune_grid

    if len(sys.argv) < 2:
        print("Usage: python -m transferability.finetune <experiment_dir>")
        print("Example: python -m transferability.finetune ./experiments/my_experiment")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    run_finetune_grid(experiment_dir)
