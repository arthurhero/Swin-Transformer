import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

imgs = torch.load('attns/images0.pt', map_location='cpu')
attns = torch.load('attns/attn0.pt', map_location='cpu')

b,_,h,w = imgs.shape

def unnormalize(img):
    mean = torch.Tensor(IMAGENET_DEFAULT_MEAN)
    std = torch.Tensor(IMAGENET_DEFAULT_STD)
    mean=(-mean/std)
    std=1./std

    img = (img-mean)/std
    return img

def display_img(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_gather(query_idx=None):
    shift = False
    shift_size = 7//2
    for i in range(len(imgs)):
        img = imgs[i] # 3 x 224 x 224
        img = img.permute(1,2,0)
        img = unnormalize(img)
        img = img.numpy()
        res = np.ones((h,2*w,3))
        res[:,:w,0]=img[:,:,2]
        res[:,:w,1]=img[:,:,1]
        res[:,:w,2]=img[:,:,0]
        for at in attns:
            for a in at: # (bxn_w) x num_heads x num_query x num_key
                attn = a.reshape(b,-1,a.shape[1],a.shape[2],a.shape[3])[i] # num_window x num_head x num_query x num_key
                num_window, num_head, num_query, num_key = attn.shape
                h_ = int(math.sqrt(num_window)) * 7
                w_ = int(math.sqrt(num_window)) * 7
                attn = attn.reshape(int(math.sqrt(num_window)),int(math.sqrt(num_window)),num_head,num_query,7, 7).permute(2,0,4,1,5,3).contiguous().reshape(-1,h_,w_,num_query) # num_head x h x w x num_query
                if shift:
                    attn = torch.roll(attn, shifts=(shift_size, shift_size), dims=(1, 2))
                print('attn shape', attn.shape)
                #attn = attn.sum(0).sum(-1) # h x w
                for idx in range(num_head):
                    res_copy = res.copy()
                    attn_copy = attn.clone()
                    if query_idx is not None:
                        attn = attn[idx,:,:,query_idx] # h x w, only choose one head
                        mask = torch.zeros(int(math.sqrt(num_window)), int(math.sqrt(num_window)), 49)
                        mask[:,:,query_idx] = 1
                        mask = mask.reshape(int(math.sqrt(num_window)), int(math.sqrt(num_window)),7,7).permute(0,2,1,3).contiguous().reshape(h_,w_)
                        if shift:
                            mask = torch.roll(mask, shifts=(shift_size, shift_size), dims=(0,1))
                        mask =  F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h,w))[0][0]
                        mask = mask.unsqueeze(2)
                        mask = mask.numpy()
                        red = np.zeros((h,w,3))
                        red[:,:,2]+=1
                        res[:,:w]=(1-mask)*res[:,:w] + mask*red
                    else:
                        attn = attn[idx].sum(-1) # h x w, only choose one head
                    attn = attn / attn.max()
                    attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=(h,w))[0][0]
                    attn = attn.unsqueeze(2)
                    attn_np = attn.numpy()
                    white = np.ones((h,w,3))
                    red = np.zeros((h,w,3))
                    red[:,:,2]+=1
                    res_attn = attn*red + (1-attn)*white
                    res[:,w:,:]=res_attn
                    attn = attn_copy
                 
                    display_img(res)
                    res = res_copy
                shift = not shift


if __name__ == '__main__':
    display_gather(48)
