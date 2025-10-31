"""Entry point for running evaluation grid as a module."""

if __name__ == "__main__":
    import sys

    from .eval_grid import run_eval_grid

    if len(sys.argv) < 2:
        print("Usage: python -m transferability.evals <experiment_dir>")
        print("Example: python -m transferability.evals ./experiments/my_experiment")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    run_eval_grid(experiment_dir)
