"""
Generate synthetic MCMC chains mimicking Planck, SH0ES, and DES posteriors.

Author: Eric D. Martin
Date: 2025-10-11
License: MIT
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def generate_planck_chain(n_samples=10000, seed=20251011):
    """
    Generate Planck-like MCMC chain with realistic correlations.

    Parameters:
    -----------
    n_samples : int
        Number of posterior samples
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame with columns: H0, Omega_m, Omega_b, n_s, tau, sigma_8
    """
    np.random.seed(seed)

    # Fiducial values from Planck 2018
    H0_mean = 67.40
    H0_std = 0.50
    Omega_m_mean = 0.315
    Omega_m_std = 0.005

    # Generate correlated samples
    # Ωm-H0 anti-correlation (physical degeneracy)
    Omega_m = np.random.normal(Omega_m_mean, Omega_m_std, n_samples)
    H0_base = np.random.normal(H0_mean, H0_std, n_samples)

    # Add Ωm-H0 correlation: lower Ωm → higher H0
    correlation_strength = 0.6
    H0 = H0_base + correlation_strength * (Omega_m_mean - Omega_m) / Omega_m_std * H0_std

    # Add systematic component (model-dependent uncertainties)
    systematic_std = 0.15
    systematic_offset = np.random.normal(0, systematic_std, n_samples)
    H0_with_sys = H0 + systematic_offset

    # Other cosmological parameters (simplified)
    Omega_b = np.random.normal(0.049, 0.001, n_samples)
    n_s = np.random.normal(0.965, 0.004, n_samples)
    tau = np.random.normal(0.054, 0.007, n_samples)
    sigma_8 = np.random.normal(0.811, 0.006, n_samples)

    chain = pd.DataFrame({
        'H0': H0_with_sys,
        'Omega_m': Omega_m,
        'Omega_b': Omega_b,
        'n_s': n_s,
        'tau': tau,
        'sigma_8': sigma_8,
        'systematic_component': systematic_offset
    })

    # Add metadata
    chain.attrs['method'] = 'CMB'
    chain.attrs['survey'] = 'Planck 2018'
    chain.attrs['H0_published'] = 67.40
    chain.attrs['H0_published_err'] = 0.50
    chain.attrs['sigma_systematic'] = np.std(systematic_offset)

    return chain


def generate_shoes_chain(n_samples=5000, seed=20251011):
    """
    Generate SH0ES-like MCMC chain with Cepheid/SN systematic structure.

    Parameters:
    -----------
    n_samples : int
        Number of posterior samples
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame with columns: H0, calibration_unc, statistical_unc, systematic_unc
    """
    np.random.seed(seed + 1)  # Offset seed

    # Fiducial values from SH0ES 2022
    H0_mean = 73.04
    H0_total_std = 1.04

    # Decompose uncertainty components
    # Statistical (Poisson, measurement noise)
    sigma_stat = 0.78
    statistical_unc = np.random.normal(0, sigma_stat, n_samples)

    # Calibration (anchor distances, zero-points)
    sigma_cal = 0.65
    calibration_unc = np.random.normal(0, sigma_cal, n_samples)

    # Systematic (metallicity, crowding, color-law)
    sigma_sys = 0.45
    systematic_unc = np.random.normal(0, sigma_sys, n_samples)

    # Total H0 with all components
    H0 = H0_mean + statistical_unc + calibration_unc + systematic_unc

    # Cepheid period-luminosity scatter
    PL_scatter = np.random.normal(0, 0.15, n_samples)

    chain = pd.DataFrame({
        'H0': H0,
        'statistical_unc': statistical_unc,
        'calibration_unc': calibration_unc,
        'systematic_unc': systematic_unc,
        'PL_scatter': PL_scatter
    })

    # Add metadata
    chain.attrs['method'] = 'Cepheid-SN'
    chain.attrs['survey'] = 'SH0ES 2022'
    chain.attrs['H0_published'] = 73.04
    chain.attrs['H0_published_err'] = 1.04
    chain.attrs['sigma_statistical'] = sigma_stat
    chain.attrs['sigma_calibration'] = sigma_cal
    chain.attrs['sigma_systematic'] = sigma_sys

    return chain


def generate_des_chain(n_samples=8000, seed=20251011):
    """
    Generate DES-like MCMC chain combining BAO and SN measurements.

    Parameters:
    -----------
    n_samples : int
        Number of posterior samples
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame with columns: H0, Omega_m, w, systematic_component
    """
    np.random.seed(seed + 2)  # Offset seed

    # Fiducial values from DES-Y5
    H0_mean = 67.19
    H0_std = 0.65
    Omega_m_mean = 0.320
    Omega_m_std = 0.012

    # Generate with BAO-Ωm correlation
    Omega_m = np.random.normal(Omega_m_mean, Omega_m_std, n_samples)
    H0_base = np.random.normal(H0_mean, H0_std, n_samples)

    # BAO constrains Ωm better than H0
    H0 = H0_base + 0.4 * (Omega_m - Omega_m_mean) / Omega_m_std * H0_std

    # Dark energy equation of state
    w = np.random.normal(-1.0, 0.05, n_samples)

    # Add systematic (covariance modeling, redshift uncertainties)
    sigma_sys = 0.25
    systematic_component = np.random.normal(0, sigma_sys, n_samples)
    H0_with_sys = H0 + systematic_component

    chain = pd.DataFrame({
        'H0': H0_with_sys,
        'Omega_m': Omega_m,
        'w': w,
        'systematic_component': systematic_component
    })

    # Add metadata
    chain.attrs['method'] = 'BAO+SN'
    chain.attrs['survey'] = 'DES-Y5 + DESI'
    chain.attrs['H0_published'] = 67.19
    chain.attrs['H0_published_err'] = 0.65
    chain.attrs['sigma_systematic'] = np.std(systematic_component)

    return chain


def compute_chain_statistics(chain, name):
    """Compute summary statistics for a chain."""
    stats = {
        'name': name,
        'n_samples': len(chain),
        'H0_mean': chain['H0'].mean(),
        'H0_std': chain['H0'].std(),
        'H0_median': chain['H0'].median(),
        'H0_q16': chain['H0'].quantile(0.16),
        'H0_q84': chain['H0'].quantile(0.84),
    }

    # Add systematic component if present
    if 'systematic_component' in chain.columns:
        stats['sigma_systematic'] = chain['systematic_component'].std()
    elif 'systematic_unc' in chain.columns:
        stats['sigma_systematic'] = chain['systematic_unc'].std()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic MCMC chains for Hubble tension analysis'
    )
    parser.add_argument('--seed', type=int, default=20251011,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='mcmc_chains/',
                       help='Output directory for chain files')
    parser.add_argument('--n-planck', type=int, default=10000,
                       help='Number of Planck samples')
    parser.add_argument('--n-shoes', type=int, default=5000,
                       help='Number of SH0ES samples')
    parser.add_argument('--n-des', type=int, default=8000,
                       help='Number of DES samples')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING SYNTHETIC MCMC CHAINS")
    print("=" * 80)
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print()

    # Generate chains
    print("Generating Planck chain...")
    planck = generate_planck_chain(args.n_planck, args.seed)
    planck_file = output_dir / 'planck_mock_chain.csv'
    planck.to_csv(planck_file, index=False)
    print(f"  ✓ Saved to {planck_file}")

    print("Generating SH0ES chain...")
    shoes = generate_shoes_chain(args.n_shoes, args.seed)
    shoes_file = output_dir / 'shoes_mock_chain.csv'
    shoes.to_csv(shoes_file, index=False)
    print(f"  ✓ Saved to {shoes_file}")

    print("Generating DES chain...")
    des = generate_des_chain(args.n_des, args.seed)
    des_file = output_dir / 'des_mock_chain.csv'
    des.to_csv(des_file, index=False)
    print(f"  ✓ Saved to {des_file}")

    # Compute and save summary statistics
    print()
    print("Computing summary statistics...")
    stats = [
        compute_chain_statistics(planck, 'Planck'),
        compute_chain_statistics(shoes, 'SH0ES'),
        compute_chain_statistics(des, 'DES')
    ]
    stats_df = pd.DataFrame(stats)
    stats_file = output_dir / 'chain_statistics.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"  ✓ Saved to {stats_file}")

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print()
    print("✓ Chain generation complete")


if __name__ == '__main__':
    main()
