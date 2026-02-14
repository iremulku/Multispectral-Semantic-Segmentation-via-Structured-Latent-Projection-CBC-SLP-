"""
fig9_like_shared_private_distributions_AUTOGEN.py

Goal (Fig. 9–style):
- Use the SAME test samples.
- Run the model under FULL mask and under each missing-modality mask.
- Extract (before fusion to decoder):
    z_shared  = shared_conv(x6_inter)
    z_pr_*    = {rgb_private_conv(x6_RGB_), nir_private_conv(x6_NIR_), swir_private_conv(x6_SWIR_)}
  and their GATED versions (multiply by mask).
- Build distribution-style figures that support your novelty:
  shared/private latent decomposition imposes an inductive bias:
    (i) shared stays stable across modality-missing patterns,
    (ii) missing modality private collapses when gated,
    (iii) energy shifts from private->shared under missing patterns,
    (iv) paired cosine(full vs missing) is higher for shared than private-concat-gated.

r≈1: almost all energy is in shared → private parts are very small.
r≈0: almost all energy is in private → shared part is very small.
r≈0.7: shared carries ~more energy than private (since denominator is shared+private).
"""

import os
import re
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==============================
# CONFIG (EDIT THESE)
# ==============================

# 1) Folder that contains your code modules: F6_CROSSVAL.py, F8_IMAGES4.py, F3_DATASET.py, mmvit4_MissingGated.py, etc.
PROJECT_DIR = r"C:\Users\Public\Server\codes\irem_DSTL_Multimodal5_LORA_latent"

# 2) Checkpoint path
MODEL_DIR = r"C:\Users\Public\Server\experiments\LORA_MULTIMODAL_OLDS\DSTL\Latentfactorizaton\2025_12_23_11_27_model0"
MODEL_FILE = "FinaliremmodelLoRA.pt"

# 3) Data / fold
TRAINSET_SIZE = 5985
FOLD_NO = 2
N_FOLDS = 5
INPUT_TYPE = "all20Ch"

# 4) Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 5) How many paired test samples to use (same samples for ALL patterns)
N_PAIRED = 600
BATCH_SIZE = 1
SHUFFLE = False

# 6) Output figures
OUT_DIR = os.path.join(PROJECT_DIR, "fig9_like_shared_private_figs")
os.makedirs(OUT_DIR, exist_ok=True)

# 7) Mask patterns (you can add more)
PATTERNS = [
    "full",
    "rgb_missing",
    "nir_missing",
    "swir_missing",
    "rgb_only",
    "nir_only",
    "swir_only",
]

MOD_NAMES = ["rgb", "nir", "swir"]  # indices 0,1,2

# 8) If you changed num_cls in your model code, set it correctly.
# Your error earlier shows checkpoint had OUT=1 for seg heads, so keep this as 1 for that ckpt.
NUM_CLS = 1


# ==============================
# Helpers: working dir + randInd auto-gen
# ==============================

def set_workdir_and_autogen_randind(project_dir: str, N: int, seed: int = 0) -> str:
    """
    CrossVal() tries to read: randInd{N}.txt from CURRENT WORKING DIRECTORY.
    We set cwd = project_dir and auto-create file if missing.
    """
    os.makedirs(project_dir, exist_ok=True)
    os.chdir(project_dir)
    path = os.path.join(project_dir, f"randInd{N}.txt")
    if os.path.exists(path):
        print("[OK] Found:", path)
        return path

    rng = np.random.RandomState(seed)
    perm = rng.permutation(N).astype(int)
    with open(path, "w") as f:
        for v in perm:
            f.write(f"{v}\n")
    print("[OK] Auto-generated:", path)
    return path


def build_mod_mask(B: int, setting: str, device: torch.device) -> torch.Tensor:
    """
    mask shape: [B,3] with values in {0,1}
    """
    mask = torch.ones(B, 3, device=device, dtype=torch.float32)
    if setting == "full":
        return mask
    if setting == "rgb_missing":
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


def gap3d(vol: torch.Tensor) -> torch.Tensor:
    """[B,C,D,H,W] -> [B,C]"""
    return vol.mean(dim=(2, 3, 4))


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    a,b: [B,C] -> cosine per sample [B]
    """
    a = a.float()
    b = b.float()
    an = a / (a.norm(dim=1, keepdim=True) + eps)
    bn = b / (b.norm(dim=1, keepdim=True) + eps)
    return (an * bn).sum(dim=1)


# ==============================
# Hook-based latent capture
# ==============================

@dataclass
class Capture:
    name: str
    tensor: Optional[torch.Tensor] = None

    def __call__(self, module, inp, out):
        self.tensor = out


class LatentHooks:
    """
    Captures model's:
      - shared_conv output        => z_shared
      - rgb_private_conv output   => z_pr_rgb (UNGATED)
      - nir_private_conv output   => z_pr_nir (UNGATED)
      - swir_private_conv output  => z_pr_swir (UNGATED)
    Then we compute gated versions ourselves using mask.
    """

    def __init__(self, model: nn.Module,
                 shared_name="shared_conv",
                 rgb_name="rgb_private_conv",
                 nir_name="nir_private_conv",
                 swir_name="swir_private_conv"):
        self.model = model
        mods = dict(model.named_modules())

        self.cap_sh = Capture(shared_name)
        self.cap_rgb = Capture(rgb_name)
        self.cap_nir = Capture(nir_name)
        self.cap_swir = Capture(swir_name)

        self.handles = []

        for cap in [self.cap_sh, self.cap_rgb, self.cap_nir, self.cap_swir]:
            if cap.name not in mods:
                raise ValueError(
                    f"Module not found for hooks: {cap.name}\n"
                    f"Tip: print([n for n,_ in model.named_modules() if 'shared' in n or 'private' in n])"
                )
            self.handles.append(mods[cap.name].register_forward_hook(cap))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def run(self, images: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          z_sh, z_pr_list (UNGATED), z_pr_g_list (GATED),
          z_pr_concat_u, z_pr_concat_g
        """
        _ = self.model(images, mask=mask)

        z_sh = self.cap_sh.tensor
        z_pr_rgb = self.cap_rgb.tensor
        z_pr_nir = self.cap_nir.tensor
        z_pr_swir = self.cap_swir.tensor

        if any(x is None for x in [z_sh, z_pr_rgb, z_pr_nir, z_pr_swir]):
            raise RuntimeError("Some latents were not captured. Check hook names.")

        B = images.size(0)
        # gate privates
        s_rgb = mask[:, 0].view(B, 1, 1, 1, 1)
        s_nir = mask[:, 1].view(B, 1, 1, 1, 1)
        s_swir = mask[:, 2].view(B, 1, 1, 1, 1)

        z_pr_g_rgb = z_pr_rgb * s_rgb
        z_pr_g_nir = z_pr_nir * s_nir
        z_pr_g_swir = z_pr_swir * s_swir

        z_pr_list = [z_pr_rgb, z_pr_nir, z_pr_swir]
        z_pr_g_list = [z_pr_g_rgb, z_pr_g_nir, z_pr_g_swir]

        z_pr_concat_u = torch.cat(z_pr_list, dim=1)   # [B, 96, 8,8,8] (32*3)
        z_pr_concat_g = torch.cat(z_pr_g_list, dim=1) # [B, 96, 8,8,8]

        return {
            "z_sh": z_sh,
            "z_pr": z_pr_list,
            "z_pr_g": z_pr_g_list,
            "z_pr_concat_u": z_pr_concat_u,
            "z_pr_concat_g": z_pr_concat_g,
        }


# ==============================
# Plotting utils (distribution-style)
# ==============================

def save_boxplot(data_by_pattern: Dict[str, np.ndarray], title: str, ylabel: str, save_path: str):
    pats = list(data_by_pattern.keys())
    vals = [data_by_pattern[p] for p in pats]
    plt.figure(figsize=(12, 5))
    plt.boxplot(vals, labels=pats, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_hist_overlay(data_by_pattern: Dict[str, np.ndarray], title: str, xlabel: str, save_path: str, bins: int = 40):
    plt.figure(figsize=(12, 6))
    for p, arr in data_by_pattern.items():
        plt.hist(arr, bins=bins, alpha=0.5, density=True, label=p)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_bar_means_with_std(data_by_pattern: Dict[str, np.ndarray], title: str, ylabel: str, save_path: str):
    pats = list(data_by_pattern.keys())
    means = [float(np.mean(data_by_pattern[p])) for p in pats]
    stds = [float(np.std(data_by_pattern[p])) for p in pats]
    plt.figure(figsize=(12, 5))
    x = np.arange(len(pats))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, pats, rotation=25, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ==============================
# MAIN
# ==============================

def main():
    print("Device:", DEVICE)

    # 0) Ensure CrossVal randInd file exists (AUTO)
    set_workdir_and_autogen_randind(PROJECT_DIR, TRAINSET_SIZE, seed=0)

    # 1) Imports (after setting cwd so relative reads work)
    from mmvit4_MissingGated import MMVit4
    from F3_DATASET import satellitedata
    from F6_CROSSVAL import CrossVal
    from F8_IMAGES4 import get_images4

    # 2) Load model
    model = MMVit4(num_cls=NUM_CLS).to(DEVICE)
    ckpt_path = os.path.join(MODEL_DIR, MODEL_FILE)
    print("Loading checkpoint:", ckpt_path)
    state = torch.load(ckpt_path, map_location=DEVICE)

    # strict load (recommended). If you REALLY changed heads, set strict=False.
    model.load_state_dict(state, strict=True)
    model.eval()

    # 3) Load data
    tsind, trind, vlind = CrossVal(TRAINSET_SIZE, FOLD_NO, N_FOLDS)
    input_images, target_masks, *_ = get_images4(
        TRAINSET_SIZE, FOLD_NO, N_FOLDS, tsind, trind, vlind, INPUT_TYPE
    )

    test_set = satellitedata(input_images[tsind], target_masks[tsind])
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    # 4) Prepare hooks
    hooks = LatentHooks(model)

    # 5) Choose the SAME paired samples (first N_PAIRED from test_loader)
    # Store them (so we can run FULL and then each pattern on the exact same tensors)
    paired_images: List[torch.Tensor] = []
    for images, _ in test_loader:
        paired_images.append(images)  # [1,3,3,224,224]
        if len(paired_images) >= N_PAIRED:
            break
    print(f"Paired samples collected: {len(paired_images)}")

    # 6) Run FULL once and store base vectors
    zsh_full_list = []
    zprG_full_list = []
    zprU_full_list = []
    # modality-wise full (for optional per-mod cos)
    zprmU_full = {mn: [] for mn in MOD_NAMES}
    zprmG_full = {mn: [] for mn in MOD_NAMES}

    for images in paired_images:
        images = images.to(DEVICE)
        B = images.size(0)
        mask_full = build_mod_mask(B, "full", DEVICE)
        lat = hooks.run(images, mask_full)

        zsh = gap3d(lat["z_sh"]).cpu()                      # [1,Csh]
        zprG = gap3d(lat["z_pr_concat_g"]).cpu()            # [1,96]
        zprU = gap3d(lat["z_pr_concat_u"]).cpu()

        zsh_full_list.append(zsh.numpy())
        zprG_full_list.append(zprG.numpy())
        zprU_full_list.append(zprU.numpy())

        for m, mn in enumerate(MOD_NAMES):
            zprmU_full[mn].append(gap3d(lat["z_pr"][m]).cpu().numpy())
            zprmG_full[mn].append(gap3d(lat["z_pr_g"][m]).cpu().numpy())

    zsh_full = np.concatenate(zsh_full_list, axis=0)     # [N,C]
    zprG_full = np.concatenate(zprG_full_list, axis=0)   # [N,96]
    zprU_full = np.concatenate(zprU_full_list, axis=0)   # [N,96]
    for mn in MOD_NAMES:
        zprmU_full[mn] = np.concatenate(zprmU_full[mn], axis=0)  # [N,32]
        zprmG_full[mn] = np.concatenate(zprmG_full[mn], axis=0)

    # 7) For each pattern, compute distributions (paired vs FULL)
    cos_shared = {}
    cos_priv_concat_g = {}
    energy_ratio_shared = {}   # ||z_sh|| / (||z_sh|| + ||z_pr_concat_g||)
    norms_sh = {}
    norms_priv_g = {}
    norms_priv_u = {}
    # per modality gated/ungated norms (to show collapse)
    norms_pr_g_mod = {mn: {} for mn in MOD_NAMES}
    norms_pr_u_mod = {mn: {} for mn in MOD_NAMES}

    for pattern in PATTERNS:
        zsh_list = []
        zprG_list = []
        zprU_list = []
        zprmU = {mn: [] for mn in MOD_NAMES}
        zprmG = {mn: [] for mn in MOD_NAMES}

        for images in paired_images:
            images = images.to(DEVICE)
            B = images.size(0)
            mask = build_mod_mask(B, pattern, DEVICE)
            lat = hooks.run(images, mask)

            zsh_list.append(gap3d(lat["z_sh"]).cpu().numpy())
            zprG_list.append(gap3d(lat["z_pr_concat_g"]).cpu().numpy())
            zprU_list.append(gap3d(lat["z_pr_concat_u"]).cpu().numpy())

            for m, mn in enumerate(MOD_NAMES):
                zprmU[mn].append(gap3d(lat["z_pr"][m]).cpu().numpy())
                zprmG[mn].append(gap3d(lat["z_pr_g"][m]).cpu().numpy())

        zsh_p = np.concatenate(zsh_list, axis=0)
        zprG_p = np.concatenate(zprG_list, axis=0)
        zprU_p = np.concatenate(zprU_list, axis=0)
        for mn in MOD_NAMES:
            zprmU[mn] = np.concatenate(zprmU[mn], axis=0)
            zprmG[mn] = np.concatenate(zprmG[mn], axis=0)

        # paired cosine wrt FULL
        cos_shared[pattern] = cosine_sim(torch.from_numpy(zsh_full), torch.from_numpy(zsh_p)).numpy()
        cos_priv_concat_g[pattern] = cosine_sim(torch.from_numpy(zprG_full), torch.from_numpy(zprG_p)).numpy()

        # norms
        nsh = np.linalg.norm(zsh_p, axis=1)
        nprg = np.linalg.norm(zprG_p, axis=1)
        npru = np.linalg.norm(zprU_p, axis=1)

        norms_sh[pattern] = nsh
        norms_priv_g[pattern] = nprg
        norms_priv_u[pattern] = npru

        # energy ratio (shared dominance when missing)
        energy_ratio_shared[pattern] = nsh / (nsh + nprg + 1e-8)

        # per modality private norms (show gating collapse precisely)
        for mn in MOD_NAMES:
            norms_pr_u_mod[mn][pattern] = np.linalg.norm(zprmU[mn], axis=1)
            norms_pr_g_mod[mn][pattern] = np.linalg.norm(zprmG[mn], axis=1)

        print(f"[{pattern}] done")

    hooks.close()

    # ==============================
    # FIGURES (Fig9-like)
    # ==============================

    # A) Paired cosine: shared vs private-concat-gated (key novelty support)
    save_boxplot(
        cos_shared,
        title="Paired cosine similarity to FULL: shared latent z_sh (higher = more stable across missing patterns)",
        ylabel="cos( z_sh(full), z_sh(pattern) )",
        save_path=os.path.join(OUT_DIR, "box_cosine_shared_vs_full_ALLpatterns.png"),
    )

    save_boxplot(
        cos_priv_concat_g,
        title="Paired cosine similarity to FULL: concatenated GATED private latents (expected to vary more with missing)",
        ylabel="cos( z_pr_g_concat(full), z_pr_g_concat(pattern) )",
        save_path=os.path.join(OUT_DIR, "box_cosine_privateG_concat_vs_full_ALLpatterns.png"),
    )

    # B) Energy ratio distribution (shared should dominate when modalities missing)
    save_hist_overlay(
        energy_ratio_shared,
        title="Distribution of shared energy ratio: ||z_sh|| / (||z_sh|| + ||z_pr_concat_g||)",
        xlabel="shared energy ratio",
        save_path=os.path.join(OUT_DIR, "hist_shared_energy_ratio_ALLpatterns.png"),
        bins=40,
    )

    save_bar_means_with_std(
        energy_ratio_shared,
        title="Mean±std shared energy ratio per pattern (shared dominance under missing patterns supports inductive bias)",
        ylabel="shared energy ratio",
        save_path=os.path.join(OUT_DIR, "bar_shared_energy_ratio_meanstd_ALLpatterns.png"),
    )

    # C) Norm distributions (shared stable, priv_g changes)
    save_boxplot(
        norms_sh,
        title="||z_sh|| distribution per pattern (shared latent magnitude should be relatively stable)",
        ylabel="||z_sh||",
        save_path=os.path.join(OUT_DIR, "box_norm_shared_ALLpatterns.png"),
    )
    save_boxplot(
        norms_priv_g,
        title="||z_pr_concat_g|| distribution per pattern (gated private magnitude should drop when modalities missing)",
        ylabel="||z_pr_concat_g||",
        save_path=os.path.join(OUT_DIR, "box_norm_privateG_concat_ALLpatterns.png"),
    )
    save_boxplot(
        norms_priv_u,
        title="||z_pr_concat_u|| distribution per pattern (UNGATED private magnitude should NOT collapse even if missing)",
        ylabel="||z_pr_concat_u||",
        save_path=os.path.join(OUT_DIR, "box_norm_privateU_concat_ALLpatterns.png"),
    )

    # D) Per-modality collapse check (most Fig9-like “modal combination effect”)
    # For each modality, show UNGATED vs GATED norms across patterns.
    for mn in MOD_NAMES:
        save_boxplot(
            norms_pr_u_mod[mn],
            title=f"UNGATED private norm distribution for modality {mn}: ||z_pr_{mn}|| (should stay non-zero)",
            ylabel=f"||z_pr_{mn}|| (UNGATED)",
            save_path=os.path.join(OUT_DIR, f"box_norm_privateU_{mn}_ALLpatterns.png"),
        )
        save_boxplot(
            norms_pr_g_mod[mn],
            title=f"GATED private norm distribution for modality {mn}: ||z~_pr_{mn}|| (should collapse when {mn} is missing)",
            ylabel=f"||z~_pr_{mn}|| (GATED)",
            save_path=os.path.join(OUT_DIR, f"box_norm_privateG_{mn}_ALLpatterns.png"),
        )

    print("\n[DONE] Saved figures to:", OUT_DIR)
    print("Key files to look at first:")
    print(" - box_cosine_shared_vs_full_ALLpatterns.png")
    print(" - box_cosine_privateG_concat_vs_full_ALLpatterns.png")
    print(" - hist_shared_energy_ratio_ALLpatterns.png")
    print(" - bar_shared_energy_ratio_meanstd_ALLpatterns.png")
    print(" - box_norm_privateG_rgb/nir/swir_ALLpatterns.png")


if __name__ == "__main__":
    main()
