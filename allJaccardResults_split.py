import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # opsiyonel ama genelde iyi oluyor
import torch 
import numpy as np
from F3_DATASET import satellitedata
from torch.utils.data import DataLoader
from F6_CROSSVAL import CrossVal
from mmvit4_MissingGated import MMVit4
from mmvit4_Missing import MMVit5
from F8_IMAGES4 import get_images4
from F5_JACCARD2 import Jaccard2, JaccardAndF1
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


MODALITY_SETTING = 'swir_only'   # 'nir_missing', 'rgb_only', vs. de yapabilirsin

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
        mask[:, 1] = 0
        mask[:, 2] = 0
    elif setting == 'nir_only':
        mask[:, 0] = 0
        mask[:, 2] = 0
    elif setting == 'swir_only':
        mask[:, 0] = 0
        mask[:, 1] = 0
    else:
        raise ValueError(f"Unknown MODALITY_SETTING: {setting}")

    return mask


# =========================================================
#   MODEL ve KAYITlı AĞI YÜKLE
# =========================================================
modelName = "FinaliremmodelLoRA.pt"   # eğitilmiş model dosyan
modelType = 'MMVit4'                 # veya 'LoRA_MMVit4', 'UNetV2', vs.
foldNo = 2
inputType = "all20Ch"              

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev)

if modelType=='MMVit4':              
    model = MMVit4().to(device)            
elif modelType=='MMVit5':              
    model = MMVit5().to(device)     

# Eğitilmiş modelin bulunduğu klasör:
modelPath = r"C:/Users/Public/Server/experiments/2026_2_3_17_49_model0"
#modelPath = r"C:/Users/Public/Server/experiments/LORA_MULTIMODAL_OLDS/DSTL/Latentfactorizaton/2025_12_23_11_27_model0"

# Ağı yükle
model.load_state_dict(torch.load(os.path.join(modelPath, modelName)))
model.eval()

# load input (for DSTL and RIT18) 

tsind,trind,vlind = CrossVal(5985,foldNo,5);
input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images4(5985, foldNo, 5, tsind, trind, vlind, inputType)


params = {'batch_size': 1, 'shuffle': False}    
test_set = satellitedata(input_images[tsind], target_masks[tsind])
test_generator = DataLoader(test_set, **params)

f1All = np.empty(test_generator.dataset.images.shape[0],dtype='float')
jcrdsAll = np.empty(test_generator.dataset.images.shape[0],dtype='float')

with torch.no_grad():
    ts = 0;
    for testim, testmas in test_generator:
        # the model
        images=testim.to(device)
        masks=testmas.to(device)
        

        if modelType in ['MMVit4', 'MMVit5']:
            mod_mask = build_mod_mask(images, MODALITY_SETTING)
            outputs = model(images, mask=mod_mask)
        else:
            outputs = model(images)


        masks = masks[:, 0, ...]   # Remove extra channel
        outputs = outputs[:, 0, ...]                     

        f1 = JaccardAndF1(torch.reshape(masks,(224*224,1)),torch.reshape(outputs,(224*224,1)))                                    
        jcrd = Jaccard2(torch.reshape(masks,(224*224,1)),torch.reshape(outputs,(224*224,1)))
        jcrdsAll[ts] = jcrd.to('cpu').numpy()[0]
        f1All[ts] = f1.to('cpu').numpy()[0]
        
       
        ts = ts+1;      

print(modelType + ", " + inputType + ", f1: ", f1All.mean() , "±" , f1All.std())
print(modelType + ", " + inputType + ", Jaccard: ", jcrdsAll.mean() , "±" , jcrdsAll.std())
            

