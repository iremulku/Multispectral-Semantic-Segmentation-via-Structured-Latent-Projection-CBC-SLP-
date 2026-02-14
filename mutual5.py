# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from F3_DATASET import satellitedata
from F6_CROSSVAL import CrossVal
from F8_IMAGES4 import get_images4

from mmvit4_MissingGated import MMVit4   # ours (shared+private)
from mmvit4_Missing import MMVit5        # base (shared-only)


# =========================================================
# CONFIG (DSTL)
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Script'in bulunduğu klasör
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Checkpoint dosyaları bu script ile aynı klasörde
OURS_CKPT_PATH = os.path.join(SCRIPT_DIR, "cbcslp.pt")
BASE_CKPT_PATH = os.path.join(SCRIPT_DIR, "cbc.pt")

# Figürler de aynı klasöre kaydedilsin
SAVE_DIR = SCRIPT_DIR
# (istersen alt klasör için: SAVE_DIR = os.path.join(SCRIPT_DIR, "outputs"); os.makedirs(SAVE_DIR, exist_ok=True))


foldNo = 2
num_folds = 5
N_SAMPLES = 5985
inputType = "all20Ch"
BATCH_SIZE = 1

# Optional caps (for memory/speed)
MAX_TRAIN_SAMPLES_FOR_PROBE = 600
MAX_TEST_SAMPLES_FOR_EVAL = None   # e.g., 300 for quick run, None for full test fold

# DSTL modality cases (training order in your scripts: RGB, NIR, SWIR)
# tensor modality idx: 0=RGB, 1=NIR, 2=SWIR
CASES = [
    "full",
    "rgb_missing",
    "nir_missing",
    "swir_missing",
    "rgb_only",
    "nir_only",
    "swir_only",
]
CASE_LABELS = ["Full", "RGB-miss", "NIR-miss", "SWIR-miss", "RGB-only", "NIR-only", "SWIR-only"]


# =========================================================
# HELPERS
# =========================================================
def build_mod_mask(images, setting: str):
    """
    DSTL tensor modality order:
      idx 0 = RGB
      idx 1 = NIR
      idx 2 = SWIR
    """
    B = images.size(0)
    mask = torch.ones(B, 3, device=images.device)

    if setting == "full":
        return mask
    elif setting == "rgb_missing":
        mask[:, 0] = 0
    elif setting == "nir_missing":
        mask[:, 1] = 0
    elif setting == "swir_missing":
        mask[:, 2] = 0
    elif setting == "rgb_only":
        mask[:, 1] = 0
        mask[:, 2] = 0
    elif setting == "nir_only":
        mask[:, 0] = 0
        mask[:, 2] = 0
    elif setting == "swir_only":
        mask[:, 0] = 0
        mask[:, 1] = 0
    else:
        raise ValueError(f"Unknown setting: {setting}")
    return mask


def bootstrap_ci_mean(values, n_boot=2000, alpha=0.05, seed=42):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = vals[idx].mean()

    mean = float(vals.mean())
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def load_model(model_class, ckpt_path):
    model = model_class().to(device)

    if not os.path.isfile(ckpt_path):
        print("SCRIPT_DIR:", SCRIPT_DIR)
        print("os.getcwd():", os.getcwd())
        print("SCRIPT_DIR içindekiler:", os.listdir(SCRIPT_DIR))
        raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    # farklı kayıt formatları için
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # DataParallel için module. prefix temizliği
    if isinstance(state, dict):
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval()
    return model



def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------- Hook utilities ----------------
class FeatureCatcher:
    def __init__(self):
        self.feature = None

    def __call__(self, module, inp, out):
        self.feature = out.detach()


def find_decoder_feature_module(model):
    """
    Pick an internal decoder feature module (avoid output heads).
    """
    named = list(model.named_modules())
    candidates = []

    for name, m in named:
        n = name.lower()

        if ("decoder_fuse" not in n) and ("decoder" not in n):
            continue

        if any(bad in n for bad in ["final", "pred", "logit", "out", "classifier"]):
            continue

        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Upsample, nn.Sequential)):
            candidates.append((name, m))

    if len(candidates) > 0:
        return candidates[-1]

    # fallback
    fallback = []
    for name, m in named:
        n = name.lower()
        if ("decoder" in n) and not any(bad in n for bad in ["final", "pred", "logit", "out", "classifier"]):
            fallback.append((name, m))
    if len(fallback) > 0:
        return fallback[-1]

    raise RuntimeError("No suitable decoder internal module found for hooking.")


def to_4d_feature(feat):
    """
    Accept:
      [B,C,H,W] -> same
      [B,M,C,H,W] -> [B,M*C,H,W]
    """
    if feat.ndim == 4:
        return feat
    if feat.ndim == 5:
        B, M, C, H, W = feat.shape
        return feat.reshape(B, M * C, H, W)
    raise RuntimeError(f"Unsupported hooked feature shape: {tuple(feat.shape)}")


def make_binary_target(masks):
    """
    masks: [B, 3, 1, H, W] in your pipeline (usually repeated)
    Use channel-0 branch as binary target.
    """
    y = masks[:, 0, ...].float()  # [B,1,H,W]
    if y.max() > 1:
        y = y / 255.0
    y = (y > 0.5).float()
    return y


# ---------------- Probe dataset/model ----------------
class FeatureDataset(Dataset):
    def __init__(self, feats, ys):
        self.feats = feats
        self.ys = ys

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, idx):
        return self.feats[idx], self.ys[idx]


class LinearProbe(nn.Module):
    """
    1x1 conv probe
    """
    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)

    def forward(self, x):
        return self.head(x)


def fit_probe(train_feats, train_y, val_feats=None, val_y=None, epochs=60, lr=1e-2, wd=1e-4, batch_size=8):
    """
    CPU training for memory safety
    """
    dev_probe = torch.device("cpu")
    model = LinearProbe(train_feats.shape[1]).to(dev_probe)

    ds = FeatureDataset(train_feats, train_y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_state = copy.deepcopy(model.state_dict())
    best_val = np.inf

    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(dev_probe)
            yb = yb.to(dev_probe)

            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="mean")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if val_feats is not None and val_y is not None:
            model.eval()
            with torch.no_grad():
                vloss = F.binary_cross_entropy_with_logits(
                    model(val_feats.to(dev_probe)), val_y.to(dev_probe), reduction="mean"
                ).item()
            if vloss < best_val:
                best_val = vloss
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model


def extract_features_and_targets(model, loader, case, hook_module_name, max_samples=None, verbose=False):
    """
    Frozen-model forward, collect hooked decoder features + targets.
    Returns CPU tensors:
      feats [N,C,H,W], ys [N,1,H,W]
    """
    target_module = dict(model.named_modules())[hook_module_name]
    catcher = FeatureCatcher()
    handle = target_module.register_forward_hook(catcher)

    feats, ys = [], []
    seen = 0
    printed = False

    try:
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                mod_mask = build_mod_mask(images, case)

                _ = model(images, mask=mod_mask)

                if catcher.feature is None:
                    raise RuntimeError(f"Hook feature is None. module={hook_module_name}")

                feat_raw = catcher.feature
                feat = to_4d_feature(feat_raw)

                y = make_binary_target(masks)
                if y.shape[-2:] != feat.shape[-2:]:
                    y = F.interpolate(y, size=feat.shape[-2:], mode="nearest")

                if verbose and (not printed):
                    print(f"[DEBUG {case}] raw hooked shape:", tuple(feat_raw.shape))
                    print(f"[DEBUG {case}] used feature shape:", tuple(feat.shape))
                    print(f"[DEBUG {case}] target shape:", tuple(y.shape))
                    print(f"[DEBUG {case}] target positive ratio:", y.mean().item())
                    printed = True

                feats.append(feat.cpu().float())
                ys.append(y.cpu().float())

                seen += images.size(0)
                if (max_samples is not None) and (seen >= max_samples):
                    break
    finally:
        handle.remove()

    feats = torch.cat(feats, dim=0)
    ys = torch.cat(ys, dim=0)

    if (max_samples is not None) and (feats.shape[0] > max_samples):
        feats = feats[:max_samples]
        ys = ys[:max_samples]

    return feats, ys


# =========================================================
# DATA PREP (DSTL fold split)
# =========================================================
def load_dstl_train_test_loaders():
    tsind, trind, vlind = CrossVal(N_SAMPLES, foldNo, num_folds)
    input_images, target_masks, *_ = get_images4(N_SAMPLES, foldNo, num_folds, tsind, trind, vlind, inputType)

    # train: trind
    train_images = input_images[trind]
    train_masks = target_masks[trind]

    # test: tsind
    test_images = input_images[tsind]
    test_masks = target_masks[tsind]

    # optional random subsampling for probe training
    if (MAX_TRAIN_SAMPLES_FOR_PROBE is not None) and (train_images.shape[0] > MAX_TRAIN_SAMPLES_FOR_PROBE):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(np.arange(train_images.shape[0]), size=MAX_TRAIN_SAMPLES_FOR_PROBE, replace=False)
        train_images = train_images[idx]
        train_masks = train_masks[idx]

    # optional random subsampling for eval
    if (MAX_TEST_SAMPLES_FOR_EVAL is not None) and (test_images.shape[0] > MAX_TEST_SAMPLES_FOR_EVAL):
        rng = np.random.default_rng(SEED + 1)
        idx = rng.choice(np.arange(test_images.shape[0]), size=MAX_TEST_SAMPLES_FOR_EVAL, replace=False)
        test_images = test_images[idx]
        test_masks = test_masks[idx]

    train_loader = DataLoader(satellitedata(train_images, train_masks), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(satellitedata(test_images, test_masks), batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples used for probe: {len(train_loader.dataset)}")
    print(f"Test samples used for eval  : {len(test_loader.dataset)}")
    return train_loader, test_loader


# =========================================================
# MAIN
# =========================================================
def main():
    train_loader, test_loader = load_dstl_train_test_loaders()

    model_ours = load_model(MMVit4, OURS_CKPT_PATH)
    model_base = load_model(MMVit5, BASE_CKPT_PATH)
    
    print("OURS ckpt:", OURS_CKPT_PATH)
    print("BASE ckpt:", BASE_CKPT_PATH)



    ours_hook_name, _ = find_decoder_feature_module(model_ours)
    base_hook_name, _ = find_decoder_feature_module(model_base)

    print("Selected hook module (ours):", ours_hook_name)
    print("Selected hook module (base):", base_hook_name)

    # scenario-wise gap list:
    # Delta_probe = BCE_base_probe - BCE_ours_probe
    case_delta = {c: [] for c in CASES}

    for ci, c in enumerate(CASES):
        print(f"\n=== Case: {c} ===")

        # 1) Extract train feats
        ours_tr_feats, ours_tr_y = extract_features_and_targets(
            model_ours, train_loader, c, ours_hook_name,
            max_samples=MAX_TRAIN_SAMPLES_FOR_PROBE,
            verbose=(ci == 0)
        )
        base_tr_feats, base_tr_y = extract_features_and_targets(
            model_base, train_loader, c, base_hook_name,
            max_samples=MAX_TRAIN_SAMPLES_FOR_PROBE,
            verbose=(ci == 0)
        )

        # 2) Extract test feats
        ours_te_feats, ours_te_y = extract_features_and_targets(
            model_ours, test_loader, c, ours_hook_name,
            max_samples=MAX_TEST_SAMPLES_FOR_EVAL,
            verbose=False
        )
        base_te_feats, base_te_y = extract_features_and_targets(
            model_base, test_loader, c, base_hook_name,
            max_samples=MAX_TEST_SAMPLES_FOR_EVAL,
            verbose=False
        )

        # 3) Fit probes
        probe_ours = fit_probe(
            train_feats=ours_tr_feats, train_y=ours_tr_y,
            val_feats=None, val_y=None,
            epochs=60, lr=1e-2, wd=1e-4, batch_size=8
        )
        probe_base = fit_probe(
            train_feats=base_tr_feats, train_y=base_tr_y,
            val_feats=None, val_y=None,
            epochs=60, lr=1e-2, wd=1e-4, batch_size=8
        )

        # 4) Evaluate sample-wise BCE on test
        with torch.no_grad():
            logits_o = probe_ours(ours_te_feats)  # CPU
            logits_b = probe_base(base_te_feats)

            bce_o = F.binary_cross_entropy_with_logits(
                logits_o, ours_te_y, reduction="none"
            ).mean(dim=(1, 2, 3)).cpu().numpy()

            bce_b = F.binary_cross_entropy_with_logits(
                logits_b, base_te_y, reduction="none"
            ).mean(dim=(1, 2, 3)).cpu().numpy()

            delta = (bce_b - bce_o).tolist()
            case_delta[c].extend(delta)

        print(f"Collected n_test={len(case_delta[c])} deltas for case={c}")

        # free memory
        del ours_tr_feats, ours_tr_y, base_tr_feats, base_tr_y
        del ours_te_feats, ours_te_y, base_te_feats, base_te_y
        del probe_ours, probe_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # summary
    means, lows, highs = [], [], []
    print("\n[DSTL RGB-NIR-SWIR] Case-wise probe-based representation gap")
    for c in CASES:
        m, lo, hi = bootstrap_ci_mean(case_delta[c], n_boot=2000, alpha=0.05, seed=SEED)
        means.append(m); lows.append(lo); highs.append(hi)
        print(f"{c:12s}  Δ_probe={m:+.6f}, 95%CI=[{lo:+.6f}, {hi:+.6f}], n={len(case_delta[c])}")

    # figure
    x = np.arange(len(CASES))
    err_low = np.array(means) - np.array(lows)
    err_hi = np.array(highs) - np.array(means)

    plt.figure(figsize=(12, 5.6))
    plt.bar(x, means, yerr=[err_low, err_hi], capsize=5,
            label=r"$\Delta_{\mathrm{probe}}(m)$ per case")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, CASE_LABELS, rotation=20)
    plt.ylabel(r"$\Delta_{\mathrm{probe}}(m)=\mathbb{E}[BCE_{base\ probe}-BCE_{ours\ probe}\mid m]$")
    plt.title("DSTL (RGB-NIR-SWIR): Representation-level information gain (single figure)\n"
              "Positive => shared+private decoder representation is more label-informative")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_png = os.path.join(SAVE_DIR, "dstl_casewise_probe_representation_gap_single_figure.png")
    out_pdf = os.path.join(SAVE_DIR, "dstl_casewise_probe_representation_gap_single_figure.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    print("\nSaved:")
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()

