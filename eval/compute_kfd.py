"""
Morphological Fréchet Distance (MFD) and Morphological Kernel Distance (MKD)
computation for skull morphology evaluation.

Usage:
    python compute_kfd.py \
        --real_female /path/to/real_female \
        --real_male /path/to/real_male \
        --gen_female /path/to/transformed_female \
        --gen_male /path/to/transformed_male \
        --checkpoint /path/to/held_out_classifier.pth \
        --pca_variance 0.95 \
        --n_bootstrap 1000 \
        --output results.json
"""
import sys

sys.path.append(".")
sys.path.append("..")

import argparse
import json
import glob
import os
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import polynomial_kernel
from tqdm import tqdm


from classifier.SeResNeXt import seresnet3d_18
from classifier.ResNet import resnet101


ARCH_REGISTRY = {
    "seresnet18": seresnet3d_18,
    "resnet101": resnet101,
}


# =============================================================================
#  Dataset
# =============================================================================

class NiftiDataset(Dataset):
    """Load NIfTI volumes from a folder."""

    def __init__(self, folder, transform=None):
        self.paths = sorted(
            glob.glob(os.path.join(folder, "*.nii.gz"))
            + glob.glob(os.path.join(folder, "*.nii"))
        )
        if len(self.paths) == 0:
            raise ValueError(f"No NIfTI files found in {folder}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = nib.load(self.paths[idx]).get_fdata().astype(np.float32)
        # Add channel dimension: (X, Y, Z) -> (1, X, Y, Z)
        img = np.expand_dims(img, axis=0)
        if self.transform:
            img = self.transform(img)
        return torch.from_numpy(img)


# =============================================================================
#  Feature extraction
# =============================================================================

def build_feature_extractor(checkpoint_path, device, arch="seresnet18"):
    """
    Load a held-out classifier checkpoint and return a feature extractor
    that outputs penultimate-layer activations (512-d after avgpool).

    :param arch: one of ARCH_REGISTRY keys; selects the constructor.
    """
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Available: {list(ARCH_REGISTRY)}")
    model = ARCH_REGISTRY[arch](in_channels=1, num_classes=1)

    state = torch.load(checkpoint_path, map_location=device)
    # Adapt key depending on your checkpoint format:
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    # Replace fc and dropout with identity to get 512-d features
    model.fc = nn.Identity()
    model.dropout = nn.Identity()

    model.eval()
    return model.to(device)


def extract_features(feature_extractor, folder, device, batch_size=1):
    """Extract penultimate-layer features for all volumes in a folder."""
    dataset = NiftiDataset(folder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting features from {Path(folder).name}"):
            batch = batch.to(device)
            feats = feature_extractor(batch)
            # Flatten spatial dims if needed: (B, C, ...) -> (B, C)
            if feats.dim() > 2:
                feats = feats.view(feats.size(0), -1)
            all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


# =============================================================================
#  MFD computation
# =============================================================================

def compute_mfd(feats_a, feats_b):
    """
    Compute the Fréchet distance between two sets of features,
    each modeled as a multivariate Gaussian.
    """
    mu_a, mu_b = feats_a.mean(axis=0), feats_b.mean(axis=0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    diff = mu_a - mu_b

    # Compute sqrt of product of covariances
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)
    # Numerical stability: discard imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    mfd = diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean)
    return float(mfd)


# =============================================================================
#  MKD computation (Morphological Kernel Distance, based on Bińkowski et al. 2018)
# =============================================================================

def compute_mkd(feats_a, feats_b, degree=3, gamma=None, coef0=1.0):
    """
    Unbiased U-statistic estimator of squared MMD with a polynomial kernel:
        k(x, y) = (gamma * <x, y> + coef0) ** degree

    Defaults follow Bińkowski et al. (2018): degree=3, gamma=1/d, coef0=1.

    Computed on raw features (no PCA): MKD does not estimate a d x d covariance
    and therefore handles the n << d regime more gracefully than MFD.

    Returns the scalar MKD value. Can go slightly negative for very close
    distributions — this is a property of the unbiased estimator, not an error.
    """
    if gamma is None:
        gamma = 1.0 / feats_a.shape[1]

    k_aa = polynomial_kernel(feats_a, feats_a, degree=degree, gamma=gamma, coef0=coef0)
    k_bb = polynomial_kernel(feats_b, feats_b, degree=degree, gamma=gamma, coef0=coef0)
    k_ab = polynomial_kernel(feats_a, feats_b, degree=degree, gamma=gamma, coef0=coef0)

    n, m = feats_a.shape[0], feats_b.shape[0]
    if n < 2 or m < 2:
        raise ValueError(
            f"compute_mkd needs at least 2 samples per set for the unbiased "
            f"estimator, got n={n}, m={m}."
        )

    # Remove diagonal terms for the unbiased estimator
    np.fill_diagonal(k_aa, 0.0)
    np.fill_diagonal(k_bb, 0.0)

    mkd = (
        k_aa.sum() / (n * (n - 1))
        + k_bb.sum() / (m * (m - 1))
        - 2.0 * k_ab.mean()
    )
    return float(mkd)


def bootstrap_mkd(feats_gen, feats_real, n_bootstrap=1000, ci=95, rng=None):
    """
    Bootstrap CIs for MKD. Symmetric to bootstrap_mfd: resamples the
    generated set with replacement and holds the real reference fixed.
    """
    if rng is None:
        rng = np.random.default_rng()
    mkd_samples = []
    n = len(feats_gen)
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping MKD"):
        idx = rng.choice(n, size=n, replace=True)
        mkd_samples.append(compute_mkd(feats_gen[idx], feats_real))

    mkd_samples = np.array(mkd_samples)
    lo = (100 - ci) / 2
    hi = 100 - lo
    return {
        "mean": float(np.mean(mkd_samples)),
        "std": float(np.std(mkd_samples)),
        "ci_low": float(np.percentile(mkd_samples, lo)),
        "ci_high": float(np.percentile(mkd_samples, hi)),
    }


# =============================================================================
#  Bootstrap confidence intervals
# =============================================================================

def bootstrap_mfd(feats_gen, feats_real, n_bootstrap=1000, ci=95, rng=None):
    """
    Bootstrap the generated set to get confidence intervals for MFD.
    The real reference set is kept fixed (it's the population we compare against).
    """
    if rng is None:
        rng = np.random.default_rng()
    mfd_samples = []
    n = len(feats_gen)
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping"):
        idx = rng.choice(n, size=n, replace=True)
        mfd_samples.append(compute_mfd(feats_gen[idx], feats_real))

    mfd_samples = np.array(mfd_samples)
    lo = (100 - ci) / 2
    hi = 100 - lo
    return {
        "mean": float(np.mean(mfd_samples)),
        "std": float(np.std(mfd_samples)),
        "ci_low": float(np.percentile(mfd_samples, lo)),
        "ci_high": float(np.percentile(mfd_samples, hi)),
    }

# =============================================================================
#  PCA sanity check
# =============================================================================

def pca_reconstruction_check(pca, feats_real, feats_gen, label_real, label_gen):
    """
    Verify that PCA reconstruction error is comparable for real and
    generated populations — confirms generated skulls lie within the
    subspace of natural variation.
    """
    proj_real = pca.transform(feats_real)
    proj_gen = pca.transform(feats_gen)
    recon_real = pca.inverse_transform(proj_real)
    recon_gen = pca.inverse_transform(proj_gen)

    err_real = np.mean(np.sum((feats_real - recon_real) ** 2, axis=1))
    err_gen = np.mean(np.sum((feats_gen - recon_gen) ** 2, axis=1))

    print(f"  PCA reconstruction error ({label_real}): {err_real:.4f}")
    print(f"  PCA reconstruction error ({label_gen}):  {err_gen:.4f}")
    print(f"  Ratio (gen/real): {err_gen / err_real:.3f}")

    return {"real": float(err_real), "generated": float(err_gen)}


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute MFD and MKD")
    parser.add_argument("--real_female", type=str, default=os.environ.get("AUTOFFS_REAL_FEMALE", "./data/real_female"), help="Folder with real female NIfTI files")
    parser.add_argument("--real_male", type=str, default=os.environ.get("AUTOFFS_REAL_MALE", "./data/real_male"), help="Folder with real male NIfTI files")
    parser.add_argument("--gen_female", type=str, required=True, help="Folder with generated female (m->f) NIfTI files")
    parser.add_argument("--gen_male", type=str, required=True, help="Folder with generated male (f->m) NIfTI files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to held-out classifier checkpoint")
    parser.add_argument("--arch", type=str, default="seresnet18", choices=list(ARCH_REGISTRY),
                        help="Architecture of the held-out classifier (must match the checkpoint).")
    parser.add_argument("--pca_variance", type=float, default=0.95, help="PCA variance to retain (default: 0.95)")
    parser.add_argument("--min_components", type=int, default=32, help="Minimum PCA components to retain (default: 32)")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for feature extraction")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "results.json"), help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Determinism setup. Independent np.random.Generator instances are spawned
    # per random block so that adding or reordering a block does not perturb
    # downstream blocks.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_seq = np.random.SeedSequence(args.seed)
    rng_intra_mfd, rng_intra_mkd, rng_boot_mfd, rng_boot_mkd = (
        np.random.default_rng(s) for s in seed_seq.spawn(4)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Feature extraction ---
    print("\n=== Loading feature extractor ===")
    feature_extractor = build_feature_extractor(args.checkpoint, device, arch=args.arch)

    print("\n=== Extracting features ===")
    feats_rf = extract_features(feature_extractor, args.real_female, device, args.batch_size)
    feats_rm = extract_features(feature_extractor, args.real_male,   device, args.batch_size)
    feats_gf = extract_features(feature_extractor, args.gen_female,  device, args.batch_size)
    feats_gm = extract_features(feature_extractor, args.gen_male,    device, args.batch_size)

    print(f"\n  Real female:      {feats_rf.shape}")
    print(f"  Real male:        {feats_rm.shape}")
    print(f"  Gen female (m→f): {feats_gf.shape}")
    print(f"  Gen male (f→m):   {feats_gm.shape}")
    print(f"  Feature dim:      {feats_rf.shape[1]}")

    # --- PCA dimensionality reduction ---
    print(f"\n=== Fitting PCA (retaining {args.pca_variance*100:.0f}% variance, min {args.min_components} components) ===")
    all_real = np.vstack([feats_rf, feats_rm])
    pca = PCA(n_components=args.pca_variance, random_state=args.seed)
    pca.fit(all_real)

    # Enforce minimum number of components
    if pca.n_components_ < args.min_components:
        print(f"  Variance-based PCA yielded only {pca.n_components_} components, re-fitting with {args.min_components}")
        pca = PCA(n_components=args.min_components, random_state=args.seed)
        pca.fit(all_real)

    print(f"  Reduced {feats_rf.shape[1]}D → {pca.n_components_}D")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Reconstruction sanity check
    print("\n=== PCA reconstruction check ===")
    recon_fem = pca_reconstruction_check(pca, feats_rf, feats_gf, "real female", "gen female")
    recon_masc = pca_reconstruction_check(pca, feats_rm, feats_gm, "real male", "gen male")

    # Project into PCA space
    rf_pca = pca.transform(feats_rf)
    rm_pca = pca.transform(feats_rm)
    gf_pca = pca.transform(feats_gf)
    gm_pca = pca.transform(feats_gm)

    # --- Compute MFD ---
    print("\n=== Computing MFD ===")
    mfd_baseline = compute_mfd(rf_pca, rm_pca)
    mfd_fem = compute_mfd(gf_pca, rf_pca)
    mfd_masc = compute_mfd(gm_pca, rm_pca)

    # Source-origin distances (overshoot check):
    # gen_female came from males, so distance to real_male reveals if transformation overshot
    mfd_fem_src = compute_mfd(gf_pca, rm_pca)
    mfd_masc_src = compute_mfd(gm_pca, rf_pca)

    print("\n  --- Primary comparisons ---")
    print(f"  MFD (real male vs real female):     {mfd_baseline:.2f}")
    print(f"  MFD (gen female vs real female):     {mfd_fem:.2f}  ({(1 - mfd_fem/mfd_baseline)*100:.1f}% reduction)")
    print(f"  MFD (gen male vs real male):         {mfd_masc:.2f}  ({(1 - mfd_masc/mfd_baseline)*100:.1f}% reduction)")

    print(f"\n  --- Source-origin comparisons (overshoot check) ---")
    print(f"  MFD (gen female vs real male):       {mfd_fem_src:.2f}  (baseline: {mfd_baseline:.2f})")
    print(f"  MFD (gen male vs real female):       {mfd_masc_src:.2f}  (baseline: {mfd_baseline:.2f})")

    # Cross-generated distance
    mfd_gen_cross = compute_mfd(gf_pca, gm_pca)
    print(f"\n  --- Cross comparisons ---")
    print(f"  MFD (gen female vs gen male):        {mfd_gen_cross:.2f}")

    # Intra-class distances via random splits (averaged over multiple splits for stability)
    n_splits = 100
    print(f"\n  --- Intra-class distances (avg over {n_splits} random splits) ---")

    mfd_intra_female = []
    for _ in range(n_splits):
        idx = rng_intra_mfd.permutation(len(rf_pca))
        half = len(idx) // 2
        mfd_intra_female.append(compute_mfd(rf_pca[idx[:half]], rf_pca[idx[half:2*half]]))
    mfd_intra_f_mean = np.mean(mfd_intra_female)
    mfd_intra_f_std = np.std(mfd_intra_female)
    print(f"  MFD (real female vs real female):    {mfd_intra_f_mean:.2f} ± {mfd_intra_f_std:.2f}")

    mfd_intra_male = []
    for _ in range(n_splits):
        idx = rng_intra_mfd.permutation(len(rm_pca))
        half = len(idx) // 2
        mfd_intra_male.append(compute_mfd(rm_pca[idx[:half]], rm_pca[idx[half:2*half]]))
    mfd_intra_m_mean = np.mean(mfd_intra_male)
    mfd_intra_m_std = np.std(mfd_intra_male)
    print(f"  MFD (real male vs real male):        {mfd_intra_m_mean:.2f} ± {mfd_intra_m_std:.2f}")

    # --- Bootstrap ---
    print(f"\n=== Bootstrap CI for MFD (B={args.n_bootstrap}) ===")
    boot_fem = bootstrap_mfd(gf_pca, rf_pca, n_bootstrap=args.n_bootstrap, rng=rng_boot_mfd)
    boot_masc = bootstrap_mfd(gm_pca, rm_pca, n_bootstrap=args.n_bootstrap, rng=rng_boot_mfd)

    print(f"  Feminization:   {boot_fem['mean']:.2f} (95% CI: {boot_fem['ci_low']:.2f}–{boot_fem['ci_high']:.2f})")
    print(f"  Masculinization: {boot_masc['mean']:.2f} (95% CI: {boot_masc['ci_low']:.2f}–{boot_masc['ci_high']:.2f})")

    # --- MKD (Morphological Kernel Distance) — cross-check on raw features ---
    # Reported alongside MFD as an unbiased, non-parametric alternative that
    # does not rely on Gaussian assumptions or covariance estimation. We use
    # raw 512-d features (no PCA) since MKD handles n << d gracefully.
    # Values are scaled ×1e3 for readability, as is standard for kernel distances.
    scale = 1e3
    print(f"\n=== Computing MKD (polynomial kernel, degree=3, on raw features; ×10³) ===")
    mkd_baseline = compute_mkd(feats_rf, feats_rm)
    mkd_fem = compute_mkd(feats_gf, feats_rf)
    mkd_masc = compute_mkd(feats_gm, feats_rm)

    mkd_fem_src = compute_mkd(feats_gf, feats_rm)
    mkd_masc_src = compute_mkd(feats_gm, feats_rf)

    print("\n  --- Primary comparisons ---")
    print(f"  MKD (real male vs real female):     {mkd_baseline*scale:.3f}")
    print(f"  MKD (gen female vs real female):     {mkd_fem*scale:.3f}  ({(1 - mkd_fem/mkd_baseline)*100:.1f}% reduction)")
    print(f"  MKD (gen male vs real male):         {mkd_masc*scale:.3f}  ({(1 - mkd_masc/mkd_baseline)*100:.1f}% reduction)")

    print(f"\n  --- Source-origin comparisons (overshoot check) ---")
    print(f"  MKD (gen female vs real male):       {mkd_fem_src*scale:.3f}  (baseline: {mkd_baseline*scale:.3f})")
    print(f"  MKD (gen male vs real female):       {mkd_masc_src*scale:.3f}  (baseline: {mkd_baseline*scale:.3f})")

    mkd_gen_cross = compute_mkd(feats_gf, feats_gm)
    print(f"\n  --- Cross comparisons ---")
    print(f"  MKD (gen female vs gen male):        {mkd_gen_cross*scale:.3f}")

    print(f"\n  --- Intra-class distances (avg over {n_splits} random splits) ---")

    mkd_intra_female = []
    for _ in range(n_splits):
        idx = rng_intra_mkd.permutation(len(feats_rf))
        half = len(idx) // 2
        mkd_intra_female.append(compute_mkd(feats_rf[idx[:half]], feats_rf[idx[half:2*half]]))
    mkd_intra_f_mean = np.mean(mkd_intra_female)
    mkd_intra_f_std = np.std(mkd_intra_female)
    print(f"  MKD (real female vs real female):    {mkd_intra_f_mean*scale:.3f} ± {mkd_intra_f_std*scale:.3f}")

    mkd_intra_male = []
    for _ in range(n_splits):
        idx = rng_intra_mkd.permutation(len(feats_rm))
        half = len(idx) // 2
        mkd_intra_male.append(compute_mkd(feats_rm[idx[:half]], feats_rm[idx[half:2*half]]))
    mkd_intra_m_mean = np.mean(mkd_intra_male)
    mkd_intra_m_std = np.std(mkd_intra_male)
    print(f"  MKD (real male vs real male):        {mkd_intra_m_mean*scale:.3f} ± {mkd_intra_m_std*scale:.3f}")

    # --- Bootstrap for MKD ---
    print(f"\n=== Bootstrap CI for MKD (B={args.n_bootstrap}) ===")
    boot_mkd_fem = bootstrap_mkd(feats_gf, feats_rf, n_bootstrap=args.n_bootstrap, rng=rng_boot_mkd)
    boot_mkd_masc = bootstrap_mkd(feats_gm, feats_rm, n_bootstrap=args.n_bootstrap, rng=rng_boot_mkd)

    print(f"  Feminization:   {boot_mkd_fem['mean']*scale:.3f} (95% CI: {boot_mkd_fem['ci_low']*scale:.3f}–{boot_mkd_fem['ci_high']*scale:.3f})")
    print(f"  Masculinization: {boot_mkd_masc['mean']*scale:.3f} (95% CI: {boot_mkd_masc['ci_low']*scale:.3f}–{boot_mkd_masc['ci_high']*scale:.3f})")
    print(f"  (Note: unbiased MKD can be slightly negative when distributions are very close.)")

    # --- Save results ---
    results = {
        "pca": {
            "original_dim": int(feats_rf.shape[1]),
            "reduced_dim": int(pca.n_components_),
            "variance_retained": float(pca.explained_variance_ratio_.sum()),
            "reconstruction_check": {
                "feminization": recon_fem,
                "masculinization": recon_masc,
            },
        },
        "mfd": {
            "real_male_vs_real_female": mfd_baseline,
            "gen_female_vs_real_female": mfd_fem,
            "gen_male_vs_real_male": mfd_masc,
            "gen_female_vs_real_male": mfd_fem_src,
            "gen_male_vs_real_female": mfd_masc_src,
            "gen_female_vs_gen_male": mfd_gen_cross,
            "intra_real_female": {"mean": mfd_intra_f_mean, "std": mfd_intra_f_std},
            "intra_real_male": {"mean": mfd_intra_m_mean, "std": mfd_intra_m_std},
            "reduction_pct_fem": (1 - mfd_fem / mfd_baseline) * 100,
            "reduction_pct_masc": (1 - mfd_masc / mfd_baseline) * 100,
            "bootstrap": {
                "feminization": boot_fem,
                "masculinization": boot_masc,
            },
        },
        "mkd": {
            "_note": "Unbiased U-statistic MKD (based on Bińkowski et al. 2018); polynomial kernel, degree=3, gamma=1/d, coef0=1. Computed on raw features (no PCA). Values are unscaled; multiply by 1e3 for the typical reporting convention.",
            "real_male_vs_real_female": mkd_baseline,
            "gen_female_vs_real_female": mkd_fem,
            "gen_male_vs_real_male": mkd_masc,
            "gen_female_vs_real_male": mkd_fem_src,
            "gen_male_vs_real_female": mkd_masc_src,
            "gen_female_vs_gen_male": mkd_gen_cross,
            "intra_real_female": {"mean": float(mkd_intra_f_mean), "std": float(mkd_intra_f_std)},
            "intra_real_male": {"mean": float(mkd_intra_m_mean), "std": float(mkd_intra_m_std)},
            "reduction_pct_fem": (1 - mkd_fem / mkd_baseline) * 100,
            "reduction_pct_masc": (1 - mkd_masc / mkd_baseline) * 100,
            "bootstrap": {
                "feminization": boot_mkd_fem,
                "masculinization": boot_mkd_masc,
            },
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Results saved to {args.output} ===")


if __name__ == "__main__":
    main()