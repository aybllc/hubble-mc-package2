# Supplementary Methodology

**Monte Carlo Calibrated Observer Tensors for Hubble Tension Resolution**

Author: Eric D. Martin
Email: eric.martin1@wsu.edu
Date: 2025-10-11

---

## Contents

1. [MCMC Chain Generation](#mcmc-chain-generation)
2. [Tensor Extraction Algorithms](#tensor-extraction-algorithms)
3. [Iterative Refinement Details](#iterative-refinement-details)
4. [Validation Procedures](#validation-procedures)
5. [Computational Requirements](#computational-requirements)

---

## MCMC Chain Generation

### Purpose

Generate synthetic MCMC posterior chains that reproduce the statistical properties of published cosmological analyses (Planck 2018, SH0ES 2022, DES-Y5).

### Method

For each probe, we generate $N$ samples from multivariate normal distributions with:
- Realistic parameter correlations (e.g., $\Omega_m$-$H_0$ degeneracy)
- Decomposed uncertainty components (statistical, calibration, systematic)
- Published mean values and uncertainties

### Planck CMB Chain

**Fiducial values:**
- $H_0 = 67.40 \pm 0.50$ km/s/Mpc
- $\Omega_m = 0.315 \pm 0.005$

**Correlation structure:**
```
H0 = H0_base + correlation_strength × (Ωm_mean - Ωm) / Ωm_std × H0_std
```
where `correlation_strength = 0.6` captures the CMB geometric degeneracy.

**Systematic component:**
- $\sigma_{\text{sys}} = 0.15$ km/s/Mpc
- Added as independent Gaussian noise to simulate model-dependent uncertainties

**Output:** 10,000 samples with columns: `H0, Omega_m, Omega_b, n_s, tau, sigma_8, systematic_component`

### SH0ES Distance Ladder Chain

**Fiducial values:**
- $H_0 = 73.04 \pm 1.04$ km/s/Mpc

**Uncertainty decomposition:**
- Statistical: $\sigma_{\text{stat}} = 0.78$ km/s/Mpc (Poisson, measurement noise)
- Calibration: $\sigma_{\text{cal}} = 0.65$ km/s/Mpc (anchor distances, zero-points)
- Systematic: $\sigma_{\text{sys}} = 0.45$ km/s/Mpc (metallicity, crowding, color-law)

**Total uncertainty:**
```
H0 = H0_mean + statistical_unc + calibration_unc + systematic_unc
```

**Output:** 5,000 samples with columns: `H0, statistical_unc, calibration_unc, systematic_unc, PL_scatter`

### DES BAO+SN Chain

**Fiducial values:**
- $H_0 = 67.19 \pm 0.65$ km/s/Mpc
- $\Omega_m = 0.320 \pm 0.012$

**Correlation structure:**
```
H0 = H0_base + 0.4 × (Ωm - Ωm_mean) / Ωm_std × H0_std
```
Weaker correlation than CMB since BAO constrains $\Omega_m$ more directly.

**Systematic component:**
- $\sigma_{\text{sys}} = 0.25$ km/s/Mpc (covariance modeling, redshift uncertainties)

**Output:** 8,000 samples with columns: `H0, Omega_m, w, systematic_component`

---

## Tensor Extraction Algorithms

### Observer Tensor Structure

An observer domain tensor has four components:
```
T_obs = [P_m, 0_t, 0_m, 0_a]
```

- **P_m**: Measurement precision weight (0 to 1 scale)
- **0_t**: Temporal calibration bias (dimensionless offset)
- **0_m**: Magnitude zero-point offset (dimensionless)
- **0_a**: Aperture/angular scale bias (dimensionless)

### Extraction Formulas

#### CMB (Planck)

High precision, model-dependent systematics:

```python
P_m = 1.0 - σ_sys / σ_H0           # Near unity due to high precision
0_t = s_sys / σ_H0                  # Temporal bias from systematic offset
0_m = corr(Ωm, H0) × 0.01          # Small coupling to matter density
0_a = -s_sys / (2 × σ_H0)          # Angular scale bias (negative convention)
```

#### Cepheid-SN (SH0ES)

Independent calibration, larger systematics:

```python
P_m = 0.75 + 0.05 × (1 - σ_sys / σ_H0)    # Lower base precision
0_t = 0.01 × tanh(s_sys)                  # Bounded temporal bias
0_m = -(σ_cal / σ_H0) × 0.05              # Calibration couples to magnitude
0_a = 0.5 + 0.1 × (s_sys / σ_H0)          # Positive aperture bias
```

#### BAO+SN (DES)

Intermediate precision, geometric calibration:

```python
P_m = 0.88 + 0.07 × (1 - σ_sys / σ_H0)   # Intermediate precision
0_t = s_sys / (1.5 × σ_H0)               # Temporal bias with damping
0_m = σ_Ωm × 0.02                        # Matter density couples to magnitude
0_a = -0.3 + 0.2 × tanh(s_sys)           # Negative base, systematic modulation
```

### Justification

These formulas are empirically calibrated to:
1. Reflect known systematic effects in each probe
2. Scale appropriately with chain statistics
3. Converge to stable values under refinement

---

## Iterative Refinement Details

### Algorithm

Starting from initial tensor extraction $\mathcal{T}^{(0)}$, we refine iteratively:

```
FOR k = 0 TO 5:
    1. Compute epistemic distance: Δ_T^(k)
    2. Merge H0 intervals using domain-aware formula
    3. Check concordance (both intervals contained?)
    4. Extract fresh tensor from chain: T_fresh^(k)
    5. Update tensor: T^(k+1) = T^(k) + α × (T_fresh^(k) - T^(k))
```

### Learning Rate

We use $\alpha = 0.15$ (15% step size) to ensure:
- Stable convergence (not too large)
- Reasonable convergence speed (not too small)

### Convergence Criteria

Convergence is achieved when:
1. $\Delta_T$ stabilizes (change < 1% per iteration)
2. Gap reaches zero (full concordance)
3. Bootstrap validation shows stability

Empirically, this occurs by iteration 5.

### Group Weighting

Early-universe group combines Planck and DES:
```
H0_early = 0.6 × H0_Planck + 0.4 × H0_DES
```
Weights based on inverse variance (Planck has higher precision).

Late-universe group uses SH0ES alone:
```
H0_late = H0_SH0ES
```

### Epistemic Distance Evolution

Expected behavior:
- **Iteration 0**: Low $\Delta_T$ (analytical tensors underestimate systematics)
- **Iterations 1-4**: Monotonic increase as refinement captures deeper systematics
- **Iteration 5**: Stabilization at true epistemic distance

---

## Validation Procedures

### Bootstrap Resampling

**Method:**
1. Take convergence trace (6 iterations)
2. Resample with replacement $n=1000$ times
3. For each resample, extract final $\Delta_T$ and gap
4. Compute mean, std, 95% confidence intervals

**Success criteria:**
- $\Delta_T$ CI should be narrow (< 5% relative width)
- Gap CI should contain zero
- Concordance success rate > 95%

### Sensitivity Analysis

**Method:**
1. Perturb final $\Delta_T$ by $\pm 5\%$ (100 trials)
2. Recompute merged interval for each perturbation
3. Check concordance robustness

**Success criteria:**
- Concordance maintained in > 95% of perturbations
- Gap remains < 0.1 km/s/Mpc in > 90% of trials

### Interval Containment Checks

**Mathematical verification:**
1. Early interval $[n_1 - u_1, n_1 + u_1]$ contained in merged $[n_m - u_m, n_m + u_m]$?
2. Late interval $[n_2 - u_2, n_2 + u_2]$ contained in merged?
3. Nominal values $n_1, n_2$ both in merged interval?

All three must be true for full concordance.

---

## Computational Requirements

### Hardware

- CPU: Any modern processor (2+ cores recommended)
- RAM: 4 GB sufficient (chains fit in memory)
- Disk: < 50 MB for all data files

### Software Dependencies

```
numpy==1.24.3
pandas==2.0.2
scipy==1.11.1
matplotlib==3.7.1  (optional, for plotting)
```

### Runtime

- Chain generation: ~5 seconds
- Tensor extraction (6 iterations): ~10 seconds
- Validation (1000 bootstrap): ~30 seconds
- **Total pipeline: < 1 minute**

### Reproducibility

Fixed random seed: `seed=20251011`

All results are deterministic given this seed.

### Verification

SHA-256 checksums provided for all output files. Users can:
1. Run `generate_all_data.py`
2. Compute checksums on generated files
3. Compare with provided `checksums/sha256sums.txt`

Match confirms exact reproduction.

---

## Limitations and Future Work

### Current Limitations

1. **Synthetic chains**: Use published statistics, not raw MCMC samples
2. **Simplified correlations**: Multivariate normal approximation
3. **Fixed learning rate**: $\alpha=0.15$ chosen empirically, not optimized

### Future Extensions

1. Apply to real MCMC chains from Planck, SH0ES, DES collaborations
2. Optimize learning rate using grid search
3. Extend to additional probes (JWST, TRGB, Mira variables)
4. Incorporate full parameter covariance matrices

### Robustness Tests

Recommended additional validation:
- Vary learning rate: $\alpha \in [0.05, 0.30]$
- Vary iteration count: $k \in [3, 10]$
- Perturb chain statistics by $\pm 1\sigma$
- Test with different random seeds

---

## References

1. **Planck Collaboration (2018).** Planck 2018 results. VI. Cosmological parameters. *Astronomy & Astrophysics*, 641, A6.

2. **Riess, A.G. et al. (2022).** A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the SH0ES Team. *ApJL*, 934, L7.

3. **DES Collaboration (2024).** Dark Energy Survey Year 5 Results: Cosmological Constraints. *In preparation*.

4. **Martin, E.D. (2025).** N/U Algebra: Conservative Uncertainty Propagation with Observer Domain Tensors. *Preprint*.

---

## Contact

Eric D. Martin
Washington State University, Vancouver
Email: eric.martin1@wsu.edu

---

*This document provides technical details for reproducing and extending the Monte Carlo calibrated observer tensor methodology.*
