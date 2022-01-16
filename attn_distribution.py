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
    cor_aq_list = list()
    cor_ad_list = list()
    layers = list()
    blocks = list()
    annos = list()
    block_count = 0
            
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
            _,num_window, num_head, num_query, num_key = at.shape
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
                else:
                    a = a.reshape(-1)
                    q = q.reshape(-1)
                    d = d.reshape(-1)

                cor_aq = torch.corrcoef(torch.stack([a,q],dim=0))[0,1].item()
                cor_ad = torch.corrcoef(torch.stack([a,d],dim=0))[0,1].item()
                cor_aq_list.append(cor_aq)
                cor_ad_list.append(cor_ad)
                layers.append(i)
                blocks.append(block_count)
                '''
                # draw the graph
                q_min = q.min().item()
                q_max = q.max().item()
                d_min = d.min().item()
                d_max = d.max().item()

                # discretize q and d
                q_grain = 100
                q_disc = (q-q_min) / (q_max-q_min)
                q_disc = (q_disc * q_grain).long()

                d_grain = 50
                d_disc = (d-d_min) / (d_max-d_min)
                d_disc = (d_disc * d_grain).long()

                a_map = torch.zeros((q_grain+1) * (d_grain+1))
                scatter_idx = q_disc*(d_grain+1) + d_disc
                scatter_mean(src = a, index = scatter_idx, dim = 0, out = a_map)
                a_map = a_map.reshape(q_grain+1, d_grain+1)

                x, y = np.meshgrid(np.linspace(d_min, d_max, d_grain+1), np.linspace(q_min, q_max, q_grain+1))
                z = a_map.numpy()
                z_min, z_max = np.abs(z).min(), np.abs(z).max()

                fig, ax = plt.subplots()
                c = ax.pcolormesh(x, y, z, cmap='Blues', vmin=z_min, vmax=z_max)
                ax.set_title('Average Attention\nlayer '+str(i)+' block '+str(j)+' head '+str(h))
                ax.axis([x.min(), x.max(), y.min(), y.max()])
                ax.set_ylabel('Normalized QK')
                ax.set_xlabel('Relative Distance')
                fig.colorbar(c, ax=ax)

                #plt.show()
                plt.savefig('attn_dist/'+'layer'+str(i)+'block'+str(j)+'head'+str(h)+'.png')
                '''
            block_count += 1
            annos.append(str(i)+', '+str(j))

    # draw the scatter plot
    cor_aq_list = np.asarray(cor_aq_list)
    cor_ad_list = np.asarray(cor_ad_list)
    fig, ax = plt.subplots()
    #scatter = ax.scatter(cor_ad_list, cor_aq_list, c=layers)
    scatter = ax.scatter(cor_ad_list, cor_aq_list, c=blocks)
    legend = ax.legend(scatter.legend_elements()[0], annos,
            title="layer, block")
    ax.add_artist(legend)
    ax.set_title('Correlation coefficient between attention and QK and distance')
    ax.set_ylabel('correlation coefficient between QK and attention')
    ax.set_xlabel('correlation coefficient between distance and attention')
    plt.show()



if __name__ == '__main__':
    gather_data()
