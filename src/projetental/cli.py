#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
import warnings
from pathlib import Path
import pandas as pd

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", message=".*least populated class.*")


def run_supervised():
    print("Running supervised experiments...")

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONWARNINGS"] = "ignore"

        result = subprocess.run(
            [sys.executable, "src/projetental/supervised_experiments.py"],
            cwd=os.getcwd(),
            env=env,
        )

        if result.returncode == 0:
            print("Supervised experiments completed successfully!")
        else:
            print(
                f"Supervised experiments failed with return code: {result.returncode}"
            )
            return False
    except Exception as e:
        print(f"Error running supervised experiments: {e}")
        return False
    return True


def run_unsupervised():
    print("Running unsupervised experiments...")

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONWARNINGS"] = "ignore"

        result = subprocess.run(
            [sys.executable, "src/projetental/unsupervised_experiment.py"],
            cwd=os.getcwd(),
            env=env,
        )

        if result.returncode == 0:
            print("Unsupervised experiments completed successfully!")
        else:
            print(
                f"Unsupervised experiments failed with return code: {result.returncode}"
            )
            return False
    except Exception as e:
        print(f"Error running unsupervised experiments: {e}")
        return False
    return True


def run_semi_supervised():
    print("Running semi-supervised experiments...")

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONWARNINGS"] = "ignore"

        result = subprocess.run(
            [sys.executable, "src/projetental/semi-supervised_experiments.py"],
            cwd=os.getcwd(),
            env=env,
        )

        if result.returncode == 0:
            print("Semi-supervised experiments completed successfully!")
        else:
            print(
                f"Semi-supervised experiments failed with return code: {result.returncode}"
            )
            return False
    except Exception as e:
        print(f"Error running semi-supervised experiments: {e}")
        return False
    return True


def process_data(xml_file, key_file, output_file):
    print("Processing data...")

    try:
        if not os.path.exists(xml_file):
            print(f"Error: {xml_file} not found")
            return False

        if not os.path.exists(key_file):
            print(f"Error: {key_file} not found")
            return False

        sys.path.append("src")
        from projetental.data_processing import (
            extract_instances_to_dataframe,
            load_gold_labels,
        )

        print(f"Processing {xml_file}...")

        df = extract_instances_to_dataframe(xml_file)
        print(f"Extracted {len(df)} instances")

        print(f"Loading gold labels from {key_file}...")
        id2sense = load_gold_labels(key_file)
        print(f"Loaded {len(id2sense)} gold labels")

        df["sense_id"] = df["target_id"].map(id2sense)

        initial_count = len(df)
        df = df.dropna(subset=["sense_id"])
        final_count = len(df)

        if initial_count > final_count:
            print(
                f"Removed {initial_count - final_count} instances without gold labels"
            )

        print(f"Saving processed dataset to {output_file}...")
        df.to_csv(output_file, index=False)

        print(f"Successfully created {output_file}")
        print(
            f"Final dataset: {len(df)} instances, {df['lemma'].nunique()} unique lemmas, {df['sense_id'].nunique()} unique senses"
        )

        return True

    except Exception as e:
        print(f"Error processing data: {e}")
        return False


def check_data_files():
    required_files = [
        "data/processed_dataset.csv",
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("Missing required data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False

    return True


def ensure_results_dir():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print(f"Results directory ready: {results_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="ProjetEntal - Word Sense Disambiguation Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --process-data corpus.xml annotations.txt dataset.csv
  %(prog)s --supervised
  %(prog)s --unsupervised
  %(prog)s --semi-supervised

Results will be saved to the 'results/' directory.
        """,
    )

    parser.add_argument(
        "--supervised", action="store_true", help="Run supervised experiments"
    )

    parser.add_argument(
        "--unsupervised", action="store_true", help="Run unsupervised experiments"
    )

    parser.add_argument(
        "--semi-supervised", action="store_true", help="Run semi-supervised experiments"
    )

    parser.add_argument(
        "--process-data",
        nargs=3,
        metavar=("XML_FILE", "KEY_FILE", "OUTPUT_FILE"),
        help="Process raw XML data files to create processed dataset",
    )

    parser.add_argument("--version", action="version", version="ProjetEntal 0.1.0")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    if args.process_data:
        xml_file, key_file, output_file = args.process_data
        if process_data(xml_file, key_file, output_file):
            print("Data processing completed successfully")
            return 0
        else:
            print("Data processing failed")
            return 1

    ensure_results_dir()

    if not check_data_files():
        print("Please ensure processed_dataset.csv exists before running experiments.")
        print("Use --process-data to create it from raw XML and key files.")
        return 1

    run_sup = args.supervised
    run_unsup = args.unsupervised
    run_semi = args.semi_supervised

    if not (run_sup or run_unsup or run_semi):
        print("Please specify which experiments to run.")
        parser.print_help()
        return 1

    results = []

    if run_sup:
        results.append(("Supervised", run_supervised()))

    if run_unsup:
        results.append(("Unsupervised", run_unsupervised()))

    if run_semi:
        results.append(("Semi-supervised", run_semi_supervised()))

    print("\nExperiment Summary:")
    all_successful = True
    for exp_name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{exp_name}: {status}")
        if not success:
            all_successful = False

    if all_successful:
        print(f"\nAll experiments completed successfully!")
        print(f"Results saved to: {Path('results').absolute()}")
        return 0
    else:
        print(f"\nSome experiments failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
