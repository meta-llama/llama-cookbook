"""Entry point for running dataset builder as a module."""

if __name__ == "__main__":
    import sys

    from .dataset_builder import run_dataset_builder

    if len(sys.argv) < 2:
        print("Usage: python -m transferability.data <experiment_dir>")
        print("Example: python -m transferability.data ./experiments/my_experiment")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    run_dataset_builder(experiment_dir)
