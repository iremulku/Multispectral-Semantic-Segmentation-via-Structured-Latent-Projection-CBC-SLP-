from __future__ import print_function
import os
import torch 
import numpy as np
import scipy.io as sio
import datetime




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
        a2 = sio.loadmat('C:/Users/Public/Server/data/DSTL/all20Ch/{}'.format(b2),verify_compressed_data_integrity=False)
        a2 = a2['inputPatch']
        input_images2.append(a2)
                        
    input_images1 = np.asarray(input_images1, dtype=np.float32)
    input_images2 = np.asarray(input_images2, dtype=np.float32)
    target_masks = np.asarray(target_masks, dtype=np.float32)
    lim=224
    
    ch9 = input_images2[:,:,:,9:10]
    ch10 = input_images2[:,:,:,10:11]
    ch11 = input_images2[:,:,:,11:12]
    
    ch12 = input_images2[:,:,:,12:13]
    ch13 = input_images2[:,:,:,13:14]
    ch14= input_images2[:,:,:,14:15]    
    
    input_images2 = np.concatenate((ch9, ch10, ch11), axis=3)
    input_images3 = np.concatenate((ch12, ch13, ch14), axis=3)
    
    input_images1 = np.reshape(input_images1[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3)) 
    input_images1 = np.moveaxis(input_images1,3,1)
    input_images2 = np.reshape(input_images2[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3)) 
    input_images2 = np.moveaxis(input_images2,3,1)
    input_images3 = np.reshape(input_images3[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3)) 
    input_images3 = np.moveaxis(input_images3,3,1)    
    target_masks = np.reshape(target_masks[0:trainSetSize*lim*lim], (trainSetSize, 1, lim, lim)) 
    
    trMeanR = input_images1[trind,0,:,:].mean()
    trMeanG = input_images1[trind,1,:,:].mean()
    trMeanB = input_images1[trind,2,:,:].mean()
    input_images1[:,0,:,:] = input_images1[:,0,:,:] - trMeanR
    input_images1[:,1,:,:] = input_images1[:,1,:,:] - trMeanG
    input_images1[:,2,:,:] = input_images1[:,2,:,:] - trMeanB
    
    trMeanN1 = input_images2[trind,0,:,:].mean()
    trMeanN2 = input_images2[trind,1,:,:].mean()
    trMeanN3 = input_images2[trind,2,:,:].mean()
    input_images2[:,0,:,:] = input_images2[:,0,:,:] - trMeanN1
    input_images2[:,1,:,:] = input_images2[:,1,:,:] - trMeanN2
    input_images2[:,2,:,:] = input_images2[:,2,:,:] - trMeanN3 
    
    trMeanS1 = input_images3[trind,0,:,:].mean()
    trMeanS2 = input_images3[trind,1,:,:].mean()
    trMeanS3 = input_images3[trind,2,:,:].mean()
    input_images3[:,0,:,:] = input_images3[:,0,:,:] - trMeanS1
    input_images3[:,1,:,:] = input_images3[:,1,:,:] - trMeanS2
    input_images3[:,2,:,:] = input_images3[:,2,:,:] - trMeanS3     
    
    
    input_images1=torch.from_numpy(input_images1)
    input_images2=torch.from_numpy(input_images2)
    input_images3=torch.from_numpy(input_images3)
    target_masks=torch.from_numpy(target_masks)
    
    input_images = torch.stack([input_images1, input_images2, input_images3], dim=1)
    target_masks = target_masks.unsqueeze(1).repeat(1, 3, 1, 1, 1)
    
    
    print("image size",input_images.shape,"mask size",target_masks.shape)
    
    print("type image",type(input_images),"type mask",type(target_masks)) 
    
    return input_images, target_masks, trMeanR, trMeanG, trMeanB