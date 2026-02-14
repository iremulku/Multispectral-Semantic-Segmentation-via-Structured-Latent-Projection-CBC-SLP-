from __future__ import print_function
import os
import torch 
import torch.nn as nn
import numpy as np
from F5_JACCARD2 import Jaccard2
from mmvit4_MissingGated import MMVit4
from mmvit4_Missing import MMVit5
import timm




# 3-modal (RGB, NIR, SWIR) için mask array
MASK_ARRAY = torch.tensor([
    [1, 0, 0],  # RGB only
    [0, 1, 0],  # NIR only
    [0, 0, 1],  # SWIR only
    [1, 1, 0],  # RGB + NIR
    [1, 0, 1],  # RGB + SWIR
    [0, 1, 1],  # NIR + SWIR
    [1, 1, 1],  # RGB + NIR + SWIR
], dtype=torch.float32)



dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#dev = torch.device("cpu")
device = torch.device(dev) 


def train_model(n_epochs, trainloss, validationloss, accuracy, model, scheduler, lrFile, training_generator, optim, lim, trainFile, trainaccFile, trainepochFile, validation_generator, valFile, valaccFile, pathm, i, modeltype):
    training_losses = []
    for epoch in range(n_epochs):
        model.train()
        batch_losses = []
        jI = 0
        totalBatches = 0
        scheduler.step()
        print('Epoch:', epoch,'LR:', scheduler.get_lr())
        lrFile.write('Epoch:'+' '+str(epoch)+' '+'LR:'+' '+str(scheduler.get_lr())+"\n")
        lrFile.write(str(scheduler.state_dict())+"\n")

        mb=0

        for trainim, trainmas in training_generator:
            mb+=1
            optim.zero_grad()
            images=trainim.to(device)
            masks=trainmas.to(device)

            
            B = images.size(0)
            
            # 1) Mask array'i GPU'ya taşı
            mask_array = MASK_ARRAY.to(images.device)
            
            # 2) Her sample için random bir index seç (0..6)
            idx = torch.randint(0, mask_array.shape[0], (B,), device=images.device)  # [B]
            
            # 3) O index'lere göre batch maske matrisi oluştur: [B,3]
            mod_mask = mask_array[idx]  # shape [B,3]
            
            # 4) Modele mask ile gönder
            outputs = model(images, mask=mod_mask)



            #outputs = model(images)
            
            if trainloss =='BCEWithLogitsLoss':
                loss=nn.BCEWithLogitsLoss()
                output = loss(outputs, masks)            
            output.backward()
            optim.step()
                        
            batch_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            if accuracy == 'Jaccard':
                masks = masks[:, 0, ...]   # Remove extra channel
                outputs = outputs[:, 0, ...] 
                thisJac = Jaccard2(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                jI = jI+thisJac.data[0]
         
            
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)
        trainFile.write(str(training_losses[epoch])+"\n")
        trainaccFile.write(str((jI/totalBatches).item())+"\n")
        trainepochFile.write(str(epoch)+"\n")
        print("Training Jaccard:",(jI/totalBatches).item()," (epoch:",epoch,")")
        lrFile.write("Training loss:"+str(training_losses[epoch])+"\n")
        lrFile.write("Training accuracy:"+str((jI/totalBatches).item())+"\n")
        
        
        torch.save(model.state_dict(), os.path.join(pathm, "iremmodel{}.pt".format(i)))
        validate(validationloss, accuracy, validation_generator, valFile, valaccFile, lim, lrFile, pathm, i, modeltype)
    torch.save(model.state_dict(), os.path.join(pathm, "FinaliremmodelLoRA.pt"))        
        
                
        
def validate(validationloss, accuracy, validation_generator, valFile, valaccFile, lim, lrFile, pathm, i, modeltype):
    jI = 0
    totalBatches = 0
    validation_losses = []
    
    
    if modeltype=='MMVit4':              
        model = MMVit4().to(device)            
    elif modeltype=='MMVit5':              
        model = MMVit5().to(device)             


    model.load_state_dict(torch.load(os.path.join(pathm, "iremmodel{}.pt".format(i))))
    model.eval()
    with torch.no_grad():
        val_losses = []
        for valim, valmas in validation_generator:
            #model.eval()
            images=valim.to(device)
            masks=valmas.to(device)

            
            B = images.size(0)
            mod_mask = torch.ones(B, 3, device=images.device)  # hepsi var
            outputs = model(images, mask=mod_mask)
         
            
            
            #outputs = model(images)

            if validationloss == 'BCEWithLogitsLoss':
                loss=nn.BCEWithLogitsLoss()
                output = loss(outputs, masks)
            val_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            if accuracy == 'Jaccard':
                masks = masks[:, 0, ...]   # Remove extra channel
                outputs = outputs[:, 0, ...] 
                thisJac = Jaccard2(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                jI = jI+thisJac.data[0] 
    dn=jI/totalBatches
    dni=dn.item()
    validation_loss = np.mean(val_losses)
    validation_losses.append(validation_loss)
    valFile.write(str(validation_losses[0])+"\n")
    valaccFile.write(str(dni)+"\n")
    print("Validation Jaccard:",dni)
    lrFile.write("Validation loss:"+str(validation_losses[0])+"\n")
    lrFile.write("Validation accuracy:"+str(dni)+"\n")
