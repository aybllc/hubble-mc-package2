"""
Master pipeline: Generate all Monte Carlo calibrated observer tensor data.

This script orchestrates the complete data generation pipeline:
1. Generate synthetic MCMC chains
2. Extract observer tensors with iterative refinement
3. Validate concordance through bootstrap resampling

Author: Eric D. Martin
Date: 2025-10-11
License: MIT
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report status."""
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n✗ ERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {description} completed successfully")
    return result


def main():
    """Run complete data generation pipeline."""
    print("=" * 80)
    print("MONTE CARLO CALIBRATED OBSERVER TENSORS - DATA GENERATION PIPELINE")
    print("=" * 80)
    print("Author: Eric D. Martin")
    print("Date: 2025-10-11")
    print("=" * 80)

    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    print(f"\nWorking directory: {script_dir}")

    # Step 1: Generate MCMC chains
    run_command(
        [sys.executable, "code/generate_chains.py",
         "--seed", "20251011",
         "--output", "mcmc_chains/",
         "--n-planck", "10000",
         "--n-shoes", "5000",
         "--n-des", "8000"],
        "Step 1: Generate MCMC Chains"
    )

    # Step 2: Extract observer tensors with iterative refinement
    run_command(
        [sys.executable, "code/extract_tensors.py",
         "--chains-dir", "mcmc_chains/",
         "--output-dir", "tensor_evolution/",
         "--n-iterations", "6",
         "--learning-rate", "0.15"],
        "Step 2: Extract Observer Tensors"
    )

    # Step 3: Validate concordance
    run_command(
        [sys.executable, "code/validate_concordance.py",
         "--convergence-file", "tensor_evolution/convergence_trace.csv",
         "--final-tensor", "tensor_evolution/tensor_evolution_iter5.json",
         "--output-dir", "validation_results/",
         "--n-bootstrap", "1000",
         "--seed", "20251011"],
        "Step 3: Validate Concordance"
    )

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  mcmc_chains/")
    print("    - planck_mock_chain.csv")
    print("    - shoes_mock_chain.csv")
    print("    - des_mock_chain.csv")
    print("    - chain_statistics.csv")
    print("\n  tensor_evolution/")
    print("    - tensor_evolution_iter0.json through iter5.json")
    print("    - convergence_trace.csv")
    print("\n  validation_results/")
    print("    - final_merged_interval.json")
    print("    - bootstrap_validation.json")
    print("    - sensitivity_analysis.json")
    print("\n✓ All data successfully generated")
    print("\nView final results:")
    print("  python -c \"import json; print(json.dumps(json.load(open('validation_results/final_merged_interval.json')), indent=2))\"")


if __name__ == '__main__':
    main()
