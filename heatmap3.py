import os
import numpy as np
import torch
import scipy.io as sio
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ---------------------------
# Your existing DSTL loader
# ---------------------------
def get_images4(trainSetSize, fno, fsiz, tsind, trind, vlind, chindex):
    input_images1=[]
    input_images2=[]
    target_masks=[]
    gettingfiles1=[]
    gettingfiles2=[]

    names1=os.listdir('C:/Users/Public/Server/data/DSTL/RGBs')
    for b1 in names1[0:trainSetSize]:
        gettingfiles1.append(b1)
        a1 = sio.loadmat('C:/Users/Public/Server/data/DSTL/RGBs/{}'.format(b1))
        a1 = a1['inputPatch']
        input_images1.append(a1)
        c=sio.loadmat('C:/Users/Public/Server/data/DSTL/class06_mats/{}'.format(b1))
        c = c['inputPatch']
        target_masks.append(c)

    names2=os.listdir('C:/Users/Public/Server/data/DSTL/all20Ch')
    for b2 in names2[0:trainSetSize]:
        gettingfiles2.append(b2)
        a2 = sio.loadmat(
            'C:/Users/Public/Server/data/DSTL/all20Ch/{}'.format(b2),
            verify_compressed_data_integrity=False
        )
        a2 = a2['inputPatch']
        input_images2.append(a2)

    input_images1 = np.asarray(input_images1, dtype=np.float32)
    input_images2 = np.asarray(input_images2, dtype=np.float32)
    target_masks  = np.asarray(target_masks, dtype=np.float32)

    lim = 224

    ch9  = input_images2[:,:,:, 9:10]
    ch10 = input_images2[:,:,:,10:11]
    ch11 = input_images2[:,:,:,11:12]
    ch12 = input_images2[:,:,:,12:13]
    ch13 = input_images2[:,:,:,13:14]
    ch14 = input_images2[:,:,:,14:15]

    input_images2 = np.concatenate((ch9, ch10, ch11), axis=3)   # modality-2
    input_images3 = np.concatenate((ch12, ch13, ch14), axis=3)  # modality-3

    input_images1 = np.reshape(input_images1[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3))
    input_images1 = np.moveaxis(input_images1, 3, 1)

    input_images2 = np.reshape(input_images2[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3))
    input_images2 = np.moveaxis(input_images2, 3, 1)

    input_images3 = np.reshape(input_images3[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3))
    input_images3 = np.moveaxis(input_images3, 3, 1)

    target_masks = np.reshape(target_masks[0:trainSetSize*lim*lim], (trainSetSize, 1, lim, lim))

    # mean subtraction (your code)
    trMeanR = input_images1[trind,0,:,:].mean()
    trMeanG = input_images1[trind,1,:,:].mean()
    trMeanB = input_images1[trind,2,:,:].mean()
    input_images1[:,0,:,:] -= trMeanR
    input_images1[:,1,:,:] -= trMeanG
    input_images1[:,2,:,:] -= trMeanB

    trMeanN1 = input_images2[trind,0,:,:].mean()
    trMeanN2 = input_images2[trind,1,:,:].mean()
    trMeanN3 = input_images2[trind,2,:,:].mean()
    input_images2[:,0,:,:] -= trMeanN1
    input_images2[:,1,:,:] -= trMeanN2
    input_images2[:,2,:,:] -= trMeanN3

    trMeanS1 = input_images3[trind,0,:,:].mean()
    trMeanS2 = input_images3[trind,1,:,:].mean()
    trMeanS3 = input_images3[trind,2,:,:].mean()
    input_images3[:,0,:,:] -= trMeanS1
    input_images3[:,1,:,:] -= trMeanS2
    input_images3[:,2,:,:] -= trMeanS3

    input_images1 = torch.from_numpy(input_images1)
    input_images2 = torch.from_numpy(input_images2)
    input_images3 = torch.from_numpy(input_images3)
    target_masks  = torch.from_numpy(target_masks)

    input_images = torch.stack([input_images1, input_images2, input_images3], dim=1)
    target_masks = target_masks.unsqueeze(1).repeat(1, 3, 1, 1, 1)

    print("image size", input_images.shape, "mask size", target_masks.shape)
    return input_images, target_masks, trMeanR, trMeanG, trMeanB


# -----------------------------------------
# PCA-1 summary for a (3,H,W) modality patch
# -----------------------------------------
def pca1_map(mod_chw: np.ndarray) -> np.ndarray:
    """
    mod_chw: (3,H,W), float
    returns: (H,W) first principal component score map
    """
    C, H, W = mod_chw.shape
    assert C == 3, "This PCA summary expects exactly 3 channels per modality."
    X = mod_chw.reshape(3, -1).T  # (H*W, 3)

    # center channels
    X = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1_dir = Vt[0]              # (3,)
    scores = X @ pc1_dir         # (H*W,)
    return scores.reshape(H, W)


# -----------------------------------------
# Absolute Spearman correlation between two 2D maps
# -----------------------------------------
def abs_spearman_2d(a: np.ndarray, b: np.ndarray) -> float:
    a1 = a.reshape(-1)
    b1 = b.reshape(-1)

    # handle degenerate cases
    if np.std(a1) < 1e-12 or np.std(b1) < 1e-12:
        return np.nan

    rho, _ = spearmanr(a1, b1)
    if np.isnan(rho):
        return np.nan

    return float(np.abs(rho))


# -----------------------------------------
# Main: compute average 3x3 correlation matrix
# -----------------------------------------
def dstl_modalitywise_heatmap(
    N=200,
    modality_names=("RGB", "NIR", "SWIR"),
    save_path="dstl_modalitywise_abs_spearman_pca1.png"
):
    trind = np.arange(N, dtype=int)

    imgs, _, _, _, _ = get_images4(
        trainSetSize=N, fno=None, fsiz=None, tsind=None,
        trind=trind, vlind=None, chindex=None
    )

    imgs = imgs.cpu().numpy()  # (N, 3 mod, 3 ch, H, W)

    mats = []
    for i in range(imgs.shape[0]):
        summaries = []
        for m in range(3):
            mod = imgs[i, m]  # (3,H,W)
            summaries.append(pca1_map(mod))

        M = np.zeros((3, 3), dtype=np.float32)
        for a in range(3):
            for b in range(3):
                M[a, b] = abs_spearman_2d(summaries[a], summaries[b])
        mats.append(M)

    mats = np.stack(mats, axis=0)  # (N,3,3)
    meanM = np.nanmean(mats, axis=0)

    # Plot heatmap: now 0..1 because absolute correlation
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(meanM, vmin=0, vmax=1)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(modality_names)
    ax.set_yticklabels(modality_names)
    ax.set_title("DSTL Modality-wise |Spearman| (PCA-1 summaries)")

    for r in range(3):
        for c in range(3):
            val = meanM[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)
    print("Mean |Spearman| matrix:\n", meanM)
    return meanM


if __name__ == "__main__":
    dstl_modalitywise_heatmap(N=200)
