import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import matplotlib.pyplot as plt
from torch_scatter import scatter_mean

imgs = torch.load('attns/images0.pt', map_location='cpu')
attns = torch.load('attns/attn0.pt', map_location='cpu')
qks = torch.load('attns/qks0.pt', map_location='cpu')
masks = torch.load('attns/masks0.pt', map_location='cpu')
distances = torch.load('attns/distances0.pt', map_location='cpu')

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

def save_img(fname,img):
    cv2.imwrite(fname,img)

def gather_data():
    for i in range(len(attns)):
        attn = attns[i]
        qk_ = qks[i]
        mask_ = masks[i]
        distance = distances[i]
        for j in range(len(attn)):
            at = attn[j] # (bxn_w) x num_heads x num_query x num_key
            qk = qk_[j] # (bxn_w) x num_heads x num_query x num_key
            mask = mask_[j] # nw x n x n 
            dis = distance[j] # n x n

            at = at.reshape(b,-1,at.shape[1],at.shape[2],at.shape[3]) # b x num_window x num_head x num_query x num_key
            _,num_window, num_head, num_query, num_key = attn.shape
            qk = qk.reshape(b,-1,num_head, num_query, num_key)

            if mask is not None:
                mask = mask.unsqueeze(0).unsqueeze(2).expand(b,-1,num_head,-1,-1).contiguous() # b x num_window x num_head x num_query x num_key
            dis = dis.reshape(1,1,1,num_query, num_key).expand(b,num_window,num_head,-1,-1).contiguous()
            
            for h in range(num_head):
                a = at[:,:,h] # b x num_window x num_query x num_key
                q = qk[:,:,h]
                if mask is not None:
                    m = mask[:,:,h]
                d = dis[:,:,h]

                if mask is not None:
                    # remove point where mask is -100
                    valid_idx = (m==0).long().nonzero(as_tuple=True)
                
                    a = a[valid_idx] # N
                    q = q[valid_idx]
                    d = d[valid_idx]

                # draw the graph
                q_min = q.min().item()
                q_max = q.max().item()
                d_min = d.min().item()
                d_max = d.max().item()

                # discretize q and d
                q_grain = 100
                q_disc = (q-q_min) / (q_max-q_min)
                q_disc = (q_disc * q_grain).long()

                d_grain = 100
                d_disc = (d-d_min) / (d_max-d_min)
                d_disc = (d_disc * d_grain).long()

                a_map = torch.zeros((q_grain+1) * (d_grain+1))
                scatter_idx = q_disc*(d_grain+1) + d_disc
                scatter_mean(src = a, index = scatter_idx, dim = 0, out = a_map)
                a_map = a_map.reshape(q_grain+1, d_grain+1)

                y, x = np.meshgrid(np.linspace(q_min, q_max, q_grain+2), np.linspace(d_min, d_max, d_grain+2))
                z = a_map.numpy()
                z_min, z_max = -np.abs(z).max(), np.abs(z).max()

                fig, ax = plt.subplots()
                c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
                ax.set_title('layer '+str(i)+' block '+str(j)+' head '+str(h))
                ax.axis([x.min(), x.max(), y.min(), y.max()])
                fig.colorbar(c, ax=ax)

                plt.show()


if __name__ == '__main__':
    gather_data()
