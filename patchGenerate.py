# =========================================================
#   DSTL TEST + SAVE RGB/NIR/SWIR + (optional) PRED/GT OVERLAY
#   - RGB is NOT saved by segplot anymore (your request).
#   - We save RGB/NIR/SWIR ourselves with robust, clean normalization.
#   - Optionally, we can still save:
#       * GT mask (mask_<ts>.png)
#       * prediction mask (pred_<ts>.png)
#       * overlay image (overlay_<ts>.png)  [simple alpha overlay, not segplot HSV trick]
#   - Preserves YOUR exact order: DataLoader(shuffle=False) and ts counter.
# =========================================================

import os
import torch
import numpy as np
import cv2

from F6_CROSSVAL import CrossVal
from F8_IMAGES4 import get_images4
from F3_DATASET import satellitedata
from torch.utils.data import DataLoader
from mmvit4_MissingGated import MMVit4
from mmvit4_Missing import MMVit5
from F5_JACCARD2 import Jaccard2, JaccardAndF1

import warnings
warnings.filterwarnings("ignore")


# =========================================================
# SETTINGS
# =========================================================
createFigures = True

# ---- MODALITE AYARI ----
MODALITY_SETTING = 'full'   # full / rgb_missing / nir_missing / swir_missing / rgb_only / nir_only / swir_only

# ---- SAVE OPTIONS ----
SAVE_RGB  = True
SAVE_NIR  = True
SAVE_SWIR = True
SAVE_GT   = True
SAVE_PRED = True
SAVE_OVERLAY = True

# If you want RGB de-meaned back using training means, set True.
# Note: robust scaling usually already looks good. This only adds back trMeanR/G/B.
RGB_ADD_BACK_TRAIN_MEAN = True

# robust scaling percentiles
P_LO, P_HI = 1, 99

# overlay alpha
OVERLAY_ALPHA = 0.45


# =========================================================
# MODALITY MASK (same as your script)
# =========================================================
def build_mod_mask(images, setting: str):
    B = images.size(0)
    mask = torch.ones(B, 3, device=images.device)

    if setting == 'full':
        return mask
    elif setting == 'rgb_missing':
        mask[:, 0] = 0
    elif setting == 'nir_missing':
        mask[:, 1] = 0
    elif setting == 'swir_missing':
        mask[:, 2] = 0
    elif setting == 'rgb_only':
        mask[:, 1] = 0; mask[:, 2] = 0
    elif setting == 'nir_only':
        mask[:, 0] = 0; mask[:, 2] = 0
    elif setting == 'swir_only':
        mask[:, 0] = 0; mask[:, 1] = 0
    else:
        raise ValueError(f"Unknown MODALITY_SETTING: {setting}")
    return mask


# =========================================================
# VIS / SAVE HELPERS
# =========================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def robust_minmax_2d(x, p_lo=1, p_hi=99):
    """x: (H,W) float -> [0,1] robust scale"""
    x = x.astype(np.float32)
    a, b = np.percentile(x, [p_lo, p_hi])
    if b - a < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1).astype(np.float32)

def robust_minmax_3ch(chw, p_lo=1, p_hi=99):
    """chw: (3,H,W) float -> (H,W,3) in [0,1] robust per-channel"""
    out = []
    for c in range(3):
        out.append(robust_minmax_2d(chw[c], p_lo, p_hi))
    return np.stack(out, axis=-1)  # H,W,3

def to_uint8_3ch(hwc01):
    """(H,W,3) [0,1] -> uint8 BGR for cv2.imwrite"""
    hwc01 = np.clip(hwc01, 0, 1)
    rgb_u8 = (255.0 * hwc01).astype(np.uint8)
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    return bgr_u8

def save_rgb_nir_swir(figuresPath, images, ts, trMeanR, trMeanG, trMeanB):
    """
    images: (1,3,3,224,224) torch tensor, modalities: 0=RGB,1=NIR,2=SWIR
    Saves:
      rgb_<ts>.png, nir_<ts>.png, swir_<ts>.png
    """
    x = images[0].detach().cpu().numpy().astype(np.float32)  # (3,3,H,W)

    # -------- RGB --------
    if SAVE_RGB:
        rgb_chw = x[0]  # (3,H,W)
        if RGB_ADD_BACK_TRAIN_MEAN:
            rgb_chw = rgb_chw.copy()
            rgb_chw[0] += trMeanR
            rgb_chw[1] += trMeanG
            rgb_chw[2] += trMeanB
        rgb_hwc01 = robust_minmax_3ch(rgb_chw, P_LO, P_HI)
        cv2.imwrite(os.path.join(figuresPath, f"rgb_{ts}.png"), to_uint8_3ch(rgb_hwc01))

    # -------- NIR --------
    if SAVE_NIR:
        nir_chw = x[1]  # (3,H,W) already mean-subtracted inside get_images4
        nir_hwc01 = robust_minmax_3ch(nir_chw, P_LO, P_HI)
        cv2.imwrite(os.path.join(figuresPath, f"nir_{ts}.png"), to_uint8_3ch(nir_hwc01))

    # -------- SWIR --------
    if SAVE_SWIR:
        swir_chw = x[2]
        swir_hwc01 = robust_minmax_3ch(swir_chw, P_LO, P_HI)
        cv2.imwrite(os.path.join(figuresPath, f"swir_{ts}.png"), to_uint8_3ch(swir_hwc01))

def save_masks_and_overlay(figuresPath, images, masks, outputs, ts, trMeanR, trMeanG, trMeanB):
    """
    Saves:
      mask_<ts>.png      (GT)
      pred_<ts>.png      (prediction)
      overlay_<ts>.png   (RGB + GT + Pred overlay)
    """
    # Prepare base RGB for overlay (same normalization as saved rgb)
    x = images[0].detach().cpu().numpy().astype(np.float32)  # (3,3,H,W)
    rgb_chw = x[0]
    if RGB_ADD_BACK_TRAIN_MEAN:
        rgb_chw = rgb_chw.copy()
        rgb_chw[0] += trMeanR
        rgb_chw[1] += trMeanG
        rgb_chw[2] += trMeanB
    rgb_hwc01 = robust_minmax_3ch(rgb_chw, P_LO, P_HI)  # (H,W,3)

    # GT + pred to (H,W) float
    gt = masks[0, 0].detach().cpu().numpy().astype(np.float32)      # (224,224)
    pr = outputs[0, 0].detach().cpu().numpy().astype(np.float32)    # (224,224)

    # If model output is logits, you might want sigmoid. If it's already [0,1], keep.
    # We'll apply sigmoid safely (won't hurt too much if already bounded).
    pr_sig = 1.0 / (1.0 + np.exp(-pr))

    # Binarize for saving masks if you want (optional); here we save grayscale probability.
    if SAVE_GT:
        cv2.imwrite(os.path.join(figuresPath, f"mask_{ts}.png"), (gt * 255).astype(np.uint8))
    if SAVE_PRED:
        cv2.imwrite(os.path.join(figuresPath, f"pred_{ts}.png"), (np.clip(pr_sig, 0, 1) * 255).astype(np.uint8))

    if SAVE_OVERLAY:
        # Create color overlays:
        # - GT in green
        # - Pred in red
        overlay = rgb_hwc01.copy()

        gt01 = np.clip(gt, 0, 1)
        pr01 = np.clip(pr_sig, 0, 1)

        # Add color hints
        color = np.zeros_like(overlay, dtype=np.float32)
        color[..., 1] = gt01          # G channel for GT
        color[..., 0] = pr01          # R channel for prediction

        overlay = (1 - OVERLAY_ALPHA) * overlay + OVERLAY_ALPHA * color
        overlay = np.clip(overlay, 0, 1)

        cv2.imwrite(os.path.join(figuresPath, f"overlay_{ts}.png"), to_uint8_3ch(overlay))


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# 'MMVit4' -> ours -> cbcslp.pt
# 'MMVit5' -> base -> cbc.pt
modelType = 'MMVit5'   # change according to your model

foldNo = 2
inputType = "all20Ch"


figuresPath = os.path.join(SCRIPT_DIR, "results", f"{modelType}_{inputType}_{foldNo}")
ensure_dir(figuresPath)


if modelType == 'MMVit4':
    model = MMVit4().to(device)
    CKPT_PATH = os.path.join(SCRIPT_DIR, "cbcslp.pt")
elif modelType == 'MMVit5':
    model = MMVit5().to(device)
    CKPT_PATH = os.path.join(SCRIPT_DIR, "cbc.pt")
else:
    raise ValueError("modelType can be only 'MMVit4' or 'MMVit5' ")


if not os.path.isfile(CKPT_PATH):
    print("SCRIPT_DIR:", SCRIPT_DIR)
    print("os.getcwd():", os.getcwd())
    print("SCRIPT_DIR içindekiler:", os.listdir(SCRIPT_DIR))
    raise FileNotFoundError(f"Checkpoint cannot be found: {CKPT_PATH}")


state = torch.load(CKPT_PATH, map_location=device)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
elif isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

if isinstance(state, dict):
    # DataParallel 'module.' 
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print("[WARN] Missing keys (first 20):", missing[:20])
if unexpected:
    print("[WARN] Unexpected keys (first 20):", unexpected[:20])

model.eval()
print("Loaded ckpt:", CKPT_PATH)
print("Figures will be saved to:", figuresPath)



# =========================================================
# LOAD DATA (same as your script)
# =========================================================
tsind, trind, vlind = CrossVal(5985, foldNo, 5)
input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images4(
    5985, foldNo, 5, tsind, trind, vlind, inputType
)

params = {'batch_size': 1, 'shuffle': False}
test_set = satellitedata(input_images[tsind], target_masks[tsind])
test_generator = DataLoader(test_set, **params)

f1All = np.empty(test_generator.dataset.images.shape[0], dtype='float')
jcrdsAll = np.empty(test_generator.dataset.images.shape[0], dtype='float')


# =========================================================
# TEST LOOP + SAVE FIGURES IN TRUE ORDER (ts counter)
# =========================================================
with torch.no_grad():
    ts = 0
    for testim, testmas in test_generator:

        images = testim.to(device)   # (1,3,3,224,224)
        masks  = testmas.to(device)  # (1,3,1,224,224)

        mod_mask = build_mod_mask(images, MODALITY_SETTING)
        outputs = model(images, mask=mod_mask)

        masks   = masks[:, 0, ...]    # (1,1,224,224)
        outputs = outputs[:, 0, ...]  # (1,1,224,224)

        f1 = JaccardAndF1(torch.reshape(masks, (224*224, 1)),
                          torch.reshape(outputs, (224*224, 1)))
        jcrd = Jaccard2(torch.reshape(masks, (224*224, 1)),
                        torch.reshape(outputs, (224*224, 1)))
        jcrdsAll[ts] = jcrd.to('cpu').numpy()[0]
        f1All[ts]    = f1.to('cpu').numpy()[0]

        if createFigures:
            # Save modalities (RGB/NIR/SWIR) with our own clean normalization
            save_rgb_nir_swir(figuresPath, images, ts, trMeanR, trMeanG, trMeanB)

            # Save GT, pred, overlay (optional)
            if SAVE_GT or SAVE_PRED or SAVE_OVERLAY:
                save_masks_and_overlay(figuresPath, images, masks, outputs, ts,
                                       trMeanR, trMeanG, trMeanB)

        ts += 1


print(modelType + ", " + inputType + ", f1: ", f1All.mean(), "±", f1All.std())
print(modelType + ", " + inputType + ", Jaccard: ", jcrdsAll.mean(), "±", jcrdsAll.std())
