import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

def img2points(img):
    '''
    convert img to points
    img - b x c x h x w
    return point position and color - b x n x 2, b x n x c
    '''
    b,c,h,w = img.shape
    pos = img.new(b,2,h,w).zero_()
    hs = torch.arange(0,h, device=img.device)
    ws = torch.arange(0,w, device=img.device)
    ys,xs = torch.meshgrid(hs,ws)
    xs=xs.unsqueeze(0).expand(b,-1,-1)
    ys=ys.unsqueeze(0).expand(b,-1,-1)
    pos[:,0]=xs
    pos[:,1]=ys
    pos = pos.view(b,2,-1).permute(0,2,1).contiguous()
    img = img.view(b,c,-1).permute(0,2,1).contiguous()
    return pos, img

def points2img(pos,pixel,h,w):
    '''
    convert points to img
    pos - b x n x 2
    pixel - b x n x c
    h,w - scalar
    return
    img - b x c x h x w
    '''
    b,n,c = pixel.shape
    img = pos.new(b,h*w,c).zero_().to(pixel.dtype)
    idx = (pos[:,:,1]*w+pos[:,:,0]).long().unsqueeze(2).expand(-1,-1,c) # b x n x c
    img.scatter_(src=pixel, index=idx, dim=1)
    return img.permute(0,2,1).reshape(b,c,h,w)

def cluster2points(cluster_pos, cluster_feat, cluster_mask, valid_row_idx, b, k, filter_invalid=False, max_size=None):
    '''
    cluster_pos - k' x m x d
    cluster_feat - k' x m x c
    cluster_mask - k' x m x 1, can be None
    valid_row_idx - can be None
    b - batch number
    k - cluster number
    return
    new_pos, new_feat, new_mask - b x n x d/c/1
    '''
    _, m, c = cluster_feat.shape
    d = cluster_pos.shape[2]
    if valid_row_idx is not None:
        new_pos = cluster_pos.new(b*k,m,d).zero_()
        new_feat = cluster_feat.new(b*k,m,c).zero_()
        new_mask = cluster_feat.new(b*k,m,1).zero_()
        new_feat[valid_row_idx] = cluster_feat
        new_pos[valid_row_idx] = cluster_pos
        if cluster_mask is None:
            new_mask[valid_row_idx] = 1
        else:
            new_mask = new_mask.to(cluster_mask.dtype)
            new_mask[valid_row_idx] = cluster_mask
    else:
        new_feat = cluster_feat
        new_pos = cluster_pos
        new_mask = cluster_mask 

    new_feat = new_feat.reshape(b,k,m,c).reshape(b,-1,c) # b x n x c
    new_pos = new_pos.reshape(b,k,m,d).reshape(b,-1,d) # b x n x d
    if new_mask is not None:
        new_mask = new_mask.reshape(b,k,m,1).reshape(b,-1,1) # b x n x 1

    if new_mask is not None and filter_invalid:
        new_mask_sum = new_mask.sum(1)
        largest_n = int(new_mask_sum.max().item()) # largest sample size
        if max_size is not None and max_size < largest_n:
            largest_n = max_size
        if largest_n == new_feat.shape[1]:
            return new_pos, new_feat, new_mask
        valid_idx = new_mask.view(-1).nonzero().squeeze() # z
        batch_idx = torch.arange(b,device=valid_idx.device).long().unsqueeze(1).expand(-1,k*m).reshape(-1)[valid_idx] # z

        valid_feat = new_feat.view(-1,c)[valid_idx] # z x c
        valid_pos = new_pos.view(-1,d)[valid_idx] # z x d
        valid_mask = new_mask.view(-1,1)[valid_idx] # z x 1
        z = len(valid_idx)
        rotate_idx = torch.arange(largest_n,device=valid_mask.device).long().repeat(int(math.ceil(z/largest_n)))[:z]
        new_pos = cluster_pos.new(b,largest_n,d).zero_()
        new_feat = cluster_feat.new(b,largest_n,c).zero_()
        new_mask = cluster_feat.new(b,largest_n,1).zero_().to(valid_mask.dtype)
        new_pos[batch_idx,rotate_idx] = valid_pos
        new_feat[batch_idx,rotate_idx] = valid_feat
        new_mask[batch_idx,rotate_idx] = valid_mask
        if new_mask.min() == 1:
            new_mask = None
    return new_pos, new_feat, new_mask

def points2cluster(pos, feat, member_idx, cluster_mask, mask=None):
    '''
    pos - b x n x d
    feat - b x n x c
    member_idx - b x k x m
    cluster_mask - b x k x m
    return
    cluster_pos, cluster_feat, cluster_mask - k' x m x d/c/1
    valid_row_idx - list of int
    '''
    b,k,m = member_idx.shape
    _,n,d = pos.shape
    if feat is not None:
        c = feat.shape[2]
    batch_tmp = torch.arange(b,device=pos.device).long().unsqueeze(1).expand(-1,k*m)
    member_idx = member_idx.view(b,-1)
    member_idx = (batch_tmp*n+member_idx).view(-1)
    cluster_pos = pos.reshape(-1,d)[member_idx].reshape(b,k,m,d).reshape(-1,m,d)
    if feat is not None:
        cluster_feat = feat.reshape(-1,c)[member_idx].reshape(b,k,m,c).reshape(-1,m,c)
    if mask is not None:
        mask = mask.reshape(-1,1)[member_idx].reshape(b,k,m,1).reshape(-1,m,1)
    # get valid cluster id
    irregular = cluster_mask is not None and cluster_mask.min() == 0
    if irregular:
        valid_row = (cluster_mask.sum(2) > 0).long() # b x k
        valid_row_idx = valid_row.view(-1).nonzero().squeeze() # z
        cluster_mask = cluster_mask.reshape(-1,m).unsqueeze(2) # k' x m x 1
        if len(valid_row_idx) == b*k:
            valid_row_idx = None
        else:
            cluster_pos = cluster_pos[valid_row_idx]
            if feat is not None:
                cluster_feat = cluster_feat[valid_row_idx]
            cluster_mask = cluster_mask[valid_row_idx]
            if mask is not None:
                mask = mask[valid_row_idx]
        cluster_pos *= cluster_mask
        if feat is not None:
            cluster_feat *= cluster_mask
        if mask is not None:
            mask *= cluster_mask
        if cluster_mask.min() > 0:
            cluster_mask = None
    else:
        cluster_mask = None
        valid_row_idx = None
    if feat is not None:
        if mask is not None:
            return cluster_pos, cluster_feat, mask, valid_row_idx
        else:
            return cluster_pos, cluster_feat, cluster_mask, valid_row_idx
    else:
        if mask is not None:
            return cluster_pos, mask, valid_row_idx
        else:
            return cluster_pos, cluster_mask, valid_row_idx

def knn_keops(query, database, k, return_dist = False, mask=None):
    '''
    get knn using pykeops library
    query - b x n x c
    database - b x N x c
    k - scalar
    mask - b x N x 1
    return
    nn_dix - b x n x k
    nn_dist - b x n x k, optinal
    '''
    from pykeops.torch import LazyTensor
    query = query.float()
    database = database.float()
    if mask is not None:
        mask = mask.float()
        max_val = database.abs().max()
        database = database * mask + (1-mask) * 10 * max_val
    b,n,c = query.shape
    N = database.shape[1]
    query_ = LazyTensor(query[:,None,:,:])
    database_ = LazyTensor(database[:,:,None,:])
    dist = ((query_-database_) ** 2).sum(-1) # b x N x n
    nn_idx = dist.argKmin(k, dim=1) # b x n x k
    if return_dist:
        nn_pos = gather_nd(database, nn_idx) # b x n x k x c
        nn_dist = ((nn_pos-query.unsqueeze(2))**2).sum(-1) # b x n x k
        return nn_idx, nn_dist
    return nn_idx

def gather_nd(inputs, nn_idx):
    '''
    inputs - b x N x c
    nn_idx - b x n x k
    return
    output - b x n x k x c
    '''
    b, N, c = inputs.shape
    _, n, k = nn_idx.shape

    batch_idx = torch.arange(b,device=inputs.device).unsqueeze(1).expand(-1,n*k).reshape(-1) # b*n*k
    nn_idx = nn_idx.reshape(-1)
    comb_idx = batch_idx * N + nn_idx
    inputs_gather = inputs.reshape(-1,c)[comb_idx]
    inputs_gather = inputs_gather.reshape(b,n,k,c) # b x n x k x c
    return inputs_gather

def batched_bincount(mat, valid_mask=None, k=None):
    '''
    batched version of torch.bincount
    mat - b x n, non-negative ints
    valid_mask - b x n, binary mask for valid points
    return counts - b x k
    '''
    b,n = mat.shape
    if k is None:
        k = mat.max().item() + 1
    result = torch.zeros(b,k,device = mat.device)
    ones = torch.ones(mat.shape,device = mat.device)
    if valid_mask is not None:
        ones *= valid_mask
    result.scatter_add_(dim=1, index=mat, src=ones) # b x k
    return result

def init_kmeanspp(points, k, pos=None, pos_lambda=None, mask=None):
    '''
    initialize kmeans++
    points - b x n x c
    k - number of means
    pos - b x n x d
    pos_lambda - weight of pos
    mask - b x n x 1
    return
    means - b x k x c
    pos_means - b x k x d
    '''
    from pykeops.torch import LazyTensor
    # get a random point
    b,n,c = points.shape
    d = pos.shape[2]
    idx = torch.randint(n,(b,)) # b
    centers = points[torch.arange(b),idx].clone() # b x c
    centers = centers.unsqueeze(1) # b x 1 x c
    if pos is not None:
        assert pos_lambda is not None, "kmeans++ pos lambda should not be None"
        pos_centers = pos[torch.arange(b),idx].clone() # b x d
        pos_centers = pos_centers.unsqueeze(1)
    if mask is not None:
        center_mask = mask[torch.arange(b),idx].clone() # b x 1
        center_mask = center_mask.unsqueeze(1)
    for _ in range(k-1):
        points_ = LazyTensor(points[:,:,None,:]) # b x n x 1 x c
        centers_ = LazyTensor(centers[:,None,:,:]) # b x 1 x k x c
        dist = ((points_ - centers_) ** 2).sum(-1) # b x n x k
        if pos is not None:
            pos_ = LazyTensor(pos[:,:,None,:])
            pos_centers_ = LazyTensor(pos_centers[:,None,:,:])
            dist_pos = ((pos_ - pos_centers_) ** 2).sum(-1)
            dist = dist + (pos_lambda / d * c) * dist_pos
        dist_min = dist.min(dim=2)
        dist_min = dist_min.squeeze(2) # b x n
        idx = (dist_min+1e-5).multinomial(num_samples=1).squeeze(1) # b
        new_center = points[torch.arange(b),idx].clone() # b x c
        new_center = new_center.unsqueeze(1) # b x 1 x c
        centers = torch.cat([centers, new_center], dim=1)
        if pos is not None:
            new_pos_center = pos[torch.arange(b),idx].clone()
            new_pos_center = new_pos_center.unsqueeze(1)
            pos_centers = torch.cat([pos_centers, new_pos_center], dim=1)
        if mask is not None:
            new_center_mask = mask[torch.arange(b),idx].clone() # b x 1
            new_center_mask = new_center_mask.unsqueeze(1)
            center_mask = torch.cat([center_mask, new_center_mask], dim=1) # b x k x 1
    if mask is not None:
        # turn invalid centers to nan
        assert center_mask.sum(1).min() > 0, "kmeans++ should have at least 1 valid point in every cluster"
        nan_mask = center_mask / center_mask
        centers *= nan_mask
        if pos is not None:
            pos_centers *= nan_mask

    if pos is not None:
        return centers, pos_centers
    else:
        return centers, None

def kmeans_keops(points, k, max_cluster_size=None, num_nearest_mean=1, num_iter=10, pos = None, pos_lambda=1, valid_mask=None, init='random', init_feat_means=None, init_pos_means=None, normalize=True):
    '''
    points - b x n x c
    k - number of means
    pos - postion of points, b x n x c
    pos_lambda - lambda of pos in dist calculation
    valid_mask - b x n x 1, binary mask indicating the valid points
    init - method of initialization, kmeans++ or random
    init_feat_means - initialize using these means, b x k x c 
    max_cluster_size - do random sampling in larger clusters; must be >= n/k
                 only affects reverse_assignment and valid_assignment_mask
    normalize - whether to normalize points to mean 0 std 1
    return
    means - b x k x c
    mean_assignment - b x n x num_nearest_mean
    reverse_assignment - b x k x m, m is the largest cluster size, invalid position filled with 0
    valid_assignment_mask - b x k x m, if sum along m gets 0, then the cluster is invalid
    '''
    #max_cluster_size=25
    points = points.detach()
    if pos is not None:
        pos = pos.detach()
    old_dtype = points.dtype
    points = points.to(torch.float32)
    from pykeops.torch import LazyTensor
    b,n,c = points.shape
    if max_cluster_size is not None:
        assert max_cluster_size>= math.ceil(n/k), "max_cluster_size should not be smaller than average"
    if pos is not None:
        d = pos.shape[2]
        pos = pos.to(points.dtype)
    # normalize mean and std
    if normalize:
        feat_mean = points.mean()
        feat_std = (points.view(-1).var(dim=0, unbiased=False)+1e-5).pow(0.5)
        points = (points-feat_mean) / feat_std
        if pos is not None:
            pos_mean = pos.mean()
            pos_std = (pos.view(-1).var(dim=0, unbiased=False)+1e-5).pow(0.5)
            pos = (pos-pos_mean) / pos_std

    if valid_mask is not None:
        valid_mask = valid_mask.detach().long()
        points *= valid_mask
        if pos is not None:
            pos *= valid_mask # make sure invalid pos and points are all 0

    # get init means
    if init_feat_means is not None:
        means = init_feat_means.detach().to(points.dtype) # b x k x c
        if normalize:
            means = (means - feat_mean) / feat_std 
        if init_pos_means is not None:
            means_pos = init_pos_means.detach().to(points.dtype)
            if normalize:
                means_pos = (means_pos - pos_mean) / pos_std 
    elif init=='random':
        rand_idx = torch.randperm(n)[:k]
        means = points[:,rand_idx,:].clone().contiguous() # b x k x c
        if pos is not None:
            means_pos = pos[:,rand_idx,:].clone().contiguous() # b x k x d
        if valid_mask is not None:
            # turn excessive invalid means to nan
            means_valid_mask = valid_mask[:,rand_idx] # b x k x 1
            row_sum = means_valid_mask.sum(1)[:,0] # b, check if all are invalid
            all_zero_index = (row_sum==0).nonzero().squeeze()
            means_valid_mask[all_zero_index, 0] = 1
            nan_mask = means_valid_mask / means_valid_mask # 1 is 1, 0 becomes nan
            means *= nan_mask
            if pos is not None:
                means_pos *= nan_mask
    elif init=='kmeans++':
        means, means_pos = init_kmeanspp(points, k, pos, pos_lambda, valid_mask) # b x k x c, b x k x d

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze(2) # b x n
    points_ = LazyTensor(points[:,:,None,:]) # b x n x 1 x c
    means_ = LazyTensor(means[:,None,:,:]) # b x 1 x k x c
    if pos is not None:
        pos_ = LazyTensor(pos[:,:,None,:]) # b x n x 1 x d
        means_pos_ = LazyTensor(means_pos[:,None,:,:]) # b x 1 x k x d

    for i in range(num_iter):
        # find nearest mean
        dist = ((points_ - means_) ** 2).sum(-1) # b x n x k
        if pos is not None:
            dist_pos = ((pos_ - means_pos_) ** 2).sum(-1) # b x n x k
            dist = dist + (pos_lambda / d * c) * dist_pos
        mean_assignment = dist.argKmin(1,dim=2).long() # b x n x 1

        # re-compute the means
        means.zero_() # content of lazytensor will change with the original tensor
        means.scatter_add_(dim=1, index=mean_assignment.expand(-1,-1,c), src=points) # invalid points will contribute 0
        bin_size = batched_bincount(mean_assignment.squeeze(2), valid_mask, k) # b x k
        means /= bin_size.unsqueeze(2)
        if pos is not None:
            means_pos.zero_()
            means_pos.scatter_add_(dim=1, index=mean_assignment.expand(-1,-1,d), src=pos)
            means_pos /= bin_size.unsqueeze(2)

    dist = ((points_ - means_) ** 2).sum(-1)
    if pos is not None:
        dist_pos = ((pos_ - means_pos_) ** 2).sum(-1) # b x n x k
        dist = dist + (pos_lambda / d * c) * dist_pos
    mean_assignment = dist.argKmin(1,dim=2).long() # b x n x 1

    max_bin_size = int(batched_bincount(mean_assignment.squeeze(2), valid_mask).max().item())
    #print("max bin size",max_bin_size, "avg size", n//k)
    if max_cluster_size is not None:
        max_bin_size = min(max_cluster_size, max_bin_size)
    # get reverse_assignment
    sorted_assignment, sorted_point_idx = mean_assignment.squeeze(2).sort(dim=-1, descending=False) # b x n, b x n
    if valid_mask is not None:
        num_valid = valid_mask.sum() # total number of valid points
        sorted_valid_mask = valid_mask.gather(index=sorted_point_idx,dim=-1) # b x n
        sorted_valid_mask = sorted_valid_mask.reshape(-1)
    sorted_assignment = sorted_assignment.reshape(-1)
    sorted_point_idx = sorted_point_idx.reshape(-1)
    if valid_mask is not None:
        sorted_valid_idx = sorted_valid_mask.nonzero().view(-1)
        sorted_assignment = sorted_assignment[sorted_valid_idx]
        sorted_point_idx = sorted_point_idx[sorted_valid_idx]
    batch_idx = torch.arange(end=b,device=mean_assignment.device).long().unsqueeze(1).expand(-1,n) # b x n
    batch_idx = batch_idx.reshape(-1)

    if valid_mask is None:
        rotate_idx = torch.arange(end=max_bin_size,device=mean_assignment.device).long().repeat(math.ceil(n/max_bin_size))[:n] # n
        rotate_idx = rotate_idx.unsqueeze(0).expand(b,-1) # b x n
        rotate_idx = rotate_idx.reshape(-1) # b*n
    else:
        batch_idx = batch_idx[sorted_valid_idx]
        rotate_idx = torch.arange(end=max_bin_size,device=mean_assignment.device).long().repeat(math.ceil(num_valid/max_bin_size))[:num_valid] # num_valid

    final_idx = (batch_idx, sorted_assignment, rotate_idx)
    reverse_assignment = torch.zeros(b, k, max_bin_size, device = mean_assignment.device).long()
    reverse_assignment -= 1 # blank space filled with -1
    reverse_assignment.index_put_(indices=final_idx, values=sorted_point_idx) # b x k x m
    valid_assignment_mask = (reverse_assignment > -1).long() # b x k x m
    reverse_assignment.clamp_(min=0)
    
    if num_nearest_mean > 1:
        mean_assignment = dist.argKmin(num_nearest_mean,dim=2).long() # b x n x num_mean

    means = means.to(old_dtype)
    del points
    if pos is not None:
        del pos

    return means, mean_assignment, reverse_assignment, valid_assignment_mask 



#if __name__ == '__main__':
