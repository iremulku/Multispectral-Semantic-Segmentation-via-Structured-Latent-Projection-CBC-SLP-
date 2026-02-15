# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# ---------------------------
# Your existing DSTL loader
# ---------------------------
def get_images4(trainSetSize, fno, fsiz, tsind, trind, vlind, chindex):
    input_images1 = []
    input_images2 = []
    target_masks = []

    names1 = os.listdir('C:/Users/Public/Server/data/DSTL/RGBs')
    names1 = sorted(names1)  # deterministic order
    for b1 in names1[0:trainSetSize]:
        a1 = sio.loadmat(f'C:/Users/Public/Server/data/DSTL/RGBs/{b1}')
        a1 = a1['inputPatch']
        input_images1.append(a1)

        c = sio.loadmat(f'C:/Users/Public/Server/data/DSTL/class06_mats/{b1}')
        c = c['inputPatch']
        target_masks.append(c)

    names2 = os.listdir('C:/Users/Public/Server/data/DSTL/all20Ch')
    names2 = sorted(names2)  # deterministic order
    for b2 in names2[0:trainSetSize]:
        a2 = sio.loadmat(
            f'C:/Users/Public/Server/data/DSTL/all20Ch/{b2}',
            verify_compressed_data_integrity=False
        )
        a2 = a2['inputPatch']
        input_images2.append(a2)

    input_images1 = np.asarray(input_images1, dtype=np.float32)
    input_images2 = np.asarray(input_images2, dtype=np.float32)
    target_masks = np.asarray(target_masks, dtype=np.float32)

    lim = 224

    # NIR-like triplet
    ch9  = input_images2[:, :, :, 9:10]
    ch10 = input_images2[:, :, :, 10:11]
    ch11 = input_images2[:, :, :, 11:12]
    # SWIR-like triplet
    ch12 = input_images2[:, :, :, 12:13]
    ch13 = input_images2[:, :, :, 13:14]
    ch14 = input_images2[:, :, :, 14:15]

    input_images2 = np.concatenate((ch9, ch10, ch11), axis=3)    # modality-2 (NIR)
    input_images3 = np.concatenate((ch12, ch13, ch14), axis=3)   # modality-3 (SWIR)

    input_images1 = np.reshape(input_images1[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3))
    input_images1 = np.moveaxis(input_images1, 3, 1)  # (N,3,H,W)

    input_images2 = np.reshape(input_images2[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3))
    input_images2 = np.moveaxis(input_images2, 3, 1)

    input_images3 = np.reshape(input_images3[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3))
    input_images3 = np.moveaxis(input_images3, 3, 1)

    target_masks = np.reshape(target_masks[0:trainSetSize*lim*lim], (trainSetSize, 1, lim, lim))

    # mean subtraction
    trMeanR = input_images1[trind, 0, :, :].mean()
    trMeanG = input_images1[trind, 1, :, :].mean()
    trMeanB = input_images1[trind, 2, :, :].mean()
    input_images1[:, 0, :, :] -= trMeanR
    input_images1[:, 1, :, :] -= trMeanG
    input_images1[:, 2, :, :] -= trMeanB

    trMeanN1 = input_images2[trind, 0, :, :].mean()
    trMeanN2 = input_images2[trind, 1, :, :].mean()
    trMeanN3 = input_images2[trind, 2, :, :].mean()
    input_images2[:, 0, :, :] -= trMeanN1
    input_images2[:, 1, :, :] -= trMeanN2
    input_images2[:, 2, :, :] -= trMeanN3

    trMeanS1 = input_images3[trind, 0, :, :].mean()
    trMeanS2 = input_images3[trind, 1, :, :].mean()
    trMeanS3 = input_images3[trind, 2, :, :].mean()
    input_images3[:, 0, :, :] -= trMeanS1
    input_images3[:, 1, :, :] -= trMeanS2
    input_images3[:, 2, :, :] -= trMeanS3

    input_images1 = torch.from_numpy(input_images1)
    input_images2 = torch.from_numpy(input_images2)
    input_images3 = torch.from_numpy(input_images3)
    target_masks = torch.from_numpy(target_masks)

    # (N,3 modalities,3 channels,H,W)
    input_images = torch.stack([input_images1, input_images2, input_images3], dim=1)
    target_masks = target_masks.unsqueeze(1).repeat(1, 3, 1, 1, 1)

    print("image size", input_images.shape, "mask size", target_masks.shape)
    return input_images, target_masks, trMeanR, trMeanG, trMeanB


# -----------------------------------------
# Modality-level feature extractor
# -----------------------------------------
def modality_feature_vector(mod_chw: np.ndarray) -> np.ndarray:
    """
    mod_chw: (3,H,W)
    Returns robust low-dim descriptor:
      [mean_c1, mean_c2, mean_c3,
       std_c1,  std_c2,  std_c3,
       eig1, eig2, eig3]
    """
    C, H, W = mod_chw.shape
    assert C == 3, "Expected exactly 3 channels per modality."
    X = mod_chw.reshape(C, -1).T  # (H*W, 3)

    means = X.mean(axis=0)
    stds = X.std(axis=0)

    Xc = X - means[None, :]
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)   # ascending
    eigvals = eigvals[::-1]             # descending

    feat = np.concatenate([means, stds, eigvals], axis=0).astype(np.float32)
    return feat


def build_tsne_dataset(imgs_np: np.ndarray):
    """
    imgs_np: (N,3 modalities,3 channels,H,W)
    Returns:
      X: (N*3, F)
      y_mod: (N*3,) -> 0:RGB,1:NIR,2:SWIR
    """
    N = imgs_np.shape[0]
    X_list = []
    y_list = []

    for i in range(N):
        for m in range(3):
            feat = modality_feature_vector(imgs_np[i, m])
            X_list.append(feat)
            y_list.append(m)

    X = np.stack(X_list, axis=0)
    y_mod = np.asarray(y_list, dtype=np.int64)
    return X, y_mod


def plot_tsne_modalities_single(
    N=200,
    perplexity=35,
    random_state=42,
    save_path=r"C:\Users\İrem ÜLKÜ\Desktop\Yeni klasör\Correlation\dstl_modalities_tsne_singleplot.png"
):
    # ensure folder exists
    out_dir = os.path.dirname(save_path)
    os.makedirs(out_dir, exist_ok=True)

    trind = np.arange(N, dtype=int)

    imgs, _, _, _, _ = get_images4(
        trainSetSize=N,
        fno=None, fsiz=None, tsind=None,
        trind=trind, vlind=None, chindex=None
    )

    imgs_np = imgs.cpu().numpy()  # (N,3,3,H,W)

    # Build feature table
    X, y_mod = build_tsne_dataset(imgs_np)

    # Standardize before t-SNE
    Xs = StandardScaler().fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
        max_iter=1500   # n_iter yerine
    )

    Z = tsne.fit_transform(Xs)  # (N*3, 2)

    # Plot
    modality_names = ["RGB", "NIR", "SWIR"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(8.6, 6.8))
    for m in range(3):
        idx = (y_mod == m)
        plt.scatter(
            Z[idx, 0], Z[idx, 1],
            s=18, alpha=0.75, label=modality_names[m],
            c=colors[m], edgecolors="none"
        )

    plt.title("Cross-Modality Embedding Distribution on DSTL")
    plt.xlabel("t-SNE embedding axis 1")
    plt.ylabel("t-SNE embedding axis 2")

    # put legend outside to avoid overlap
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", save_path)

    # quick numeric summary: centroid distances
    centroids = []
    for m in range(3):
        idx = (y_mod == m)
        centroids.append(Z[idx].mean(axis=0))
    centroids = np.stack(centroids, axis=0)

    def dist(a, b):
        return float(np.linalg.norm(centroids[a] - centroids[b]))

    print("Centroid distances:")
    print(f"RGB-NIR : {dist(0,1):.4f}")
    print(f"RGB-SWIR: {dist(0,2):.4f}")
    print(f"NIR-SWIR: {dist(1,2):.4f}")


if __name__ == "__main__":
    plot_tsne_modalities_single(
        N=200,
        perplexity=35,
        random_state=42,
        save_path=r"C:\Users\İrem ÜLKÜ\Desktop\Yeni klasör\Correlation\dstl_modalities_tsne_singleplot.png"
    )


