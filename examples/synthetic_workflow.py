"""Run after installing the package: python examples/synthetic_workflow.py."""
from dlunmix.cli import main

if __name__ == "__main__":
    main(["demo", "--out", "synthetic-output"])
