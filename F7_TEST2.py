from __future__ import print_function
import torch 
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from F5_JACCARD2 import Jaccard2
from mmvit4_MissingGated import MMVit4
from mmvit4_Missing import MMVit5
from F11_SEGPLOT import segplot
import time
import torch
import numpy as np




dev = "cuda:0"  
device = torch.device(dev) 

# ---- MODALITE AYARI: BURAYI DEĞİŞTİREREK FARKLI SENARYO TEST EDECEKSİN ----
# seçenek örnekleri:
# 'full'          : RGB + NIR + SWIR (hiçbir modalite eksik değil)
# 'rgb_missing'   : NIR + SWIR var, RGB eksik
# 'nir_missing'   : RGB + SWIR var, NIR eksik
# 'swir_missing'  : RGB + NIR var, SWIR eksik
# 'rgb_only'      : sadece RGB
# 'nir_only'      : sadece NIR
# 'swir_only'     : sadece SWIR
MODALITY_SETTING = 'full'   # ← burayı her run öncesi değiştirirsin

def build_mod_mask(images, setting: str):
    """
    images: [B, 3, D, H, W]
    setting: yukarıdaki MODALITY_SETTING stringlerinden biri
    return: [B, 3] mask tensor (1 = var, 0 = eksik)
    """
    B = images.size(0)
    mask = torch.ones(B, 3, device=images.device)

    if setting == 'full':
        # hepsi var → [1,1,1]
        return mask

    elif setting == 'rgb_missing':
        mask[:, 0] = 0      # RGB eksik

    elif setting == 'nir_missing':
        mask[:, 1] = 0      # NIR eksik

    elif setting == 'swir_missing':
        mask[:, 2] = 0      # SWIR eksik

    elif setting == 'rgb_only':
        mask[:, 1] = 0      # NIR kapalı
        mask[:, 2] = 0      # SWIR kapalı

    elif setting == 'nir_only':
        mask[:, 0] = 0      # RGB kapalı
        mask[:, 2] = 0      # SWIR kapalı

    elif setting == 'swir_only':
        mask[:, 0] = 0      # RGB kapalı
        mask[:, 1] = 0      # NIR kapalı

    else:
        raise ValueError(f"Unknown MODALITY_SETTING: {setting}")

    return mask


def test_model(test_generator, lim, testFile, testaccFile, i, modeltype, pathm, trMeanR, trMeanG, trMeanB):
    
    data_folder = os.path.join("../../experiments")

    if modeltype=='MMVit4':              
        net = MMVit4().to(device)            
    elif modeltype=='MMVit5':              
        net = MMVit5().to(device)           

 
        
    net.load_state_dict(torch.load(os.path.join(pathm, "FinaliremmodelLoRA.pt")))

    jI = 0
    totalBatches = 0
    test_losses = []
    net.eval()
    with torch.no_grad():
        t_losses = []
        t=0
        start_time = time.time()
        for testim, testmas in test_generator:
            images=testim.to(device)
            masks=testmas.to(device)
            

            if modeltype in ['MMVit4', 'MMVit5']:
                mod_mask = build_mod_mask(images, MODALITY_SETTING)
                outputs = net(images, mask=mod_mask)
            else:
                outputs = net(images)



            #outputs = model(images)

            if t==0:
                fig=plt.figure()
                axes=[]
                images2 = images[:, 0, ...]
                fimage=images2[0].permute(1, 2, 0)
                fimage[:,:,0]=(images2[0][0,:,:])
                fimage[:,:,1]=(images2[0][1,:,:])
                fimage[:,:,2]=(images2[0][2,:,:])
                fimage=fimage.cpu().numpy()
                axes.append(fig.add_subplot(1, 2, 1))
                outputs2 = outputs[:, 0, ...] 
                foutput=outputs2[0].permute(1, 2, 0)
                foutput=foutput.cpu().numpy()
                plt.imshow(np.squeeze(foutput, axis=2),  cmap='gray')
                subplot_title=("Test Predicted Mask")
                axes[-1].set_title(subplot_title)
                axes.append(fig.add_subplot(1, 2, 2))
                masks2 = masks[:, 0, ...]
                fmask=masks2[0].permute(1, 2, 0)
                fmask=fmask.cpu().numpy()
                plt.imshow(np.squeeze(fmask, axis=2),  cmap='gray')
                subplot_title=("Ground Truth Mask")
                axes[-1].set_title(subplot_title)
                n_curve = 'mask_comparison.png'
                plt.savefig(os.path.join(pathm, n_curve))
                plt.show()
                segplot(pathm, lim, fimage, foutput, fmask,  trMeanR, trMeanG, trMeanB)
            losst=nn.BCEWithLogitsLoss()
            output = losst(outputs, masks)
            t_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            masks = masks[:, 0, ...]   # Remove extra channel
            outputs = outputs[:, 0, ...] 
            thisJac = Jaccard2(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
            jI = jI+thisJac.data[0]
            t+=1
   
    dn=jI/totalBatches
    dni=dn.item()
    test_loss = np.mean(t_losses)
    test_losses.append(test_loss)
    testFile.write(str(test_losses[0])+"\n")
    testaccFile.write(str(dni)+"\n")
    print("Test Jaccard:",dni)

