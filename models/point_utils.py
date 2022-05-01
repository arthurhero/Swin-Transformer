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
    '''
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
    '''
    new_feat = cluster_feat
    new_pos = cluster_pos
    new_mask = cluster_mask 

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
        '''
        else:
            cluster_pos = cluster_pos[valid_row_idx]
            if feat is not None:
                cluster_feat = cluster_feat[valid_row_idx]
            cluster_mask = cluster_mask[valid_row_idx]
            if mask is not None:
                mask = mask[valid_row_idx]
        '''
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

def kmeans(points, k, max_cluster_size=None, num_nearest_mean=1, num_iter=10, pos = None, pos_lambda=1, valid_mask=None, init='random', init_feat_means=None, init_pos_means=None, normalize=True, balanced=False, strictly_balanced=False, fillup=False):
    '''
    points - b x n x c
    k - number of means
    pos - postion of points, b x n x c
    pos_lambda - lambda of pos in dist calculation, can be scalar or a list with length k
    valid_mask - b x n x 1, binary mask indicating the valid points
    init - method of initialization, kmeans++ or random
    init_feat_means - initialize using these means, b x k x c 
    max_cluster_size - do random sampling in larger clusters; must be >= n/k
                 only affects reverse_assignment and valid_assignment_mask
    normalize - whether to normalize points to mean 0 std 1
    balanced - try to balance the cluster sizes
    return
    means - b x k x c
    mean_assignment - b x n x num_nearest_mean
    reverse_assignment - b x k x m, m is the largest cluster size, invalid position filled with 0
    valid_assignment_mask - b x k x m, if sum along m gets 0, then the cluster is invalid
    '''
    start1 = time.time()
    #max_cluster_size=25
    points = points.detach()
    if pos is not None:
        pos = pos.detach()
    old_dtype = points.dtype
    points = points.to(torch.float32)
    points = torch.randn(points.shape,device=points.device,dtype=torch.float32)
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

    means[means.isnan().nonzero(as_tuple=True)]=float('inf')
    if pos is not None:
        means_pos[means_pos.isnan().nonzero(as_tuple=True)]=float('inf')

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
            dist += (pos_lambda / d * c) * dist_pos
            #dist = dist_pos
        mean_assignment = dist.argKmin(1,dim=2).long() # b x n x 1

        # re-compute the means
        means.zero_() # content of lazytensor will change with the original tensor
        means.scatter_add_(dim=1, index=mean_assignment.expand(-1,-1,c), src=points) # invalid points will contribute 0
        bin_size = batched_bincount(mean_assignment.squeeze(2), valid_mask, k) # b x k
        means /= bin_size.unsqueeze(2)
        means[means.isnan().nonzero(as_tuple=True)]=float('inf')
        if pos is not None:
            means_pos.zero_()
            means_pos.scatter_add_(dim=1, index=mean_assignment.expand(-1,-1,d), src=pos)
            means_pos /= bin_size.unsqueeze(2)
            means_pos[means_pos.isnan().nonzero(as_tuple=True)]=float('inf')
        if balanced:
            largest_idx=bin_size.argmax(dim=1) # b
            smallest_idx=bin_size.argmin(dim=1) # b
            bidx = torch.arange(b,device=points.device).long()
            means[bidx,smallest_idx] = means[bidx,largest_idx].clone() + torch.randn(b,c,device=points.device)/100.0
            if pos is not None:
                means_pos[bidx,smallest_idx] = means_pos[bidx,largest_idx].clone() + torch.randn(b,d,device=points.device) / 100.0


    if not strictly_balanced:
        inf_bidx, inf_kidx = means[:,:,0].isinf().nonzero(as_tuple=True)
        dist = ((points_ - means_) ** 2).sum(-1) # b x n x k
        if pos is not None:
            dist_pos = ((pos_ - means_pos_) ** 2).sum(-1) # b x n x k
            dist += (pos_lambda / d * c) * dist_pos
            #dist = dist_pos
        mean_assignment = dist.argKmin(1,dim=2).long() # b x n x 1
        if fillup:
            mutual_choice = torch.zeros(b,n,k,device=points.device)
            mutual_choice.scatter_(index=mean_assignment, dim=2, src=torch.ones(b,n,1,device=points.device))
            member_grab = dist.argKmin(max_cluster_size,dim=1).permute(0,2,1) # b x msc x k
            mc_tmp = mutual_choice.clone()
            mc_tmp.scatter_(index=member_grab, dim=1, src=torch.ones(b,max_cluster_size,k,device=points.device))
            added_points = mc_tmp - mutual_choice # b x n x k

            bin_size = mutual_choice.sum(1) # b x k 
            #assert bin_size.max() <= max_cluster_size * 2, "max bin size is over 2*msc!!"
            added_num = added_points.sum(1) # b x k 
            del_num = (added_num-(max_cluster_size-bin_size).clamp(min=0)).clamp(min=0)

            add_src, added_points_idx = added_points.topk(added_num.max().long(),dim=1,sorted=True) # b x a x k
            add_src2 = torch.arange(add_src.shape[1],device=add_src.device).reshape(1,-1,1).expand(b,-1,k) # b x a x k
            add_src2 = (add_src2 >= del_num.unsqueeze(1)).to(add_src.dtype)
            add_src *= add_src2
            added_points.scatter_(index = added_points_idx, dim=1, src=add_src)
            mutual_choice += added_points
            mutual_choice.clamp_(max=1)
            mutual_choice[inf_bidx,:,inf_kidx]=0
            mutual_choice[inf_bidx,:max_cluster_size,inf_kidx]=1
            
            #mutual_choice.scatter_(index=member_grab, dim=1, src=torch.ones(b,max_cluster_size,k,device=points.device))
            bin_size = mutual_choice.sum(1) # b x k 
            # split clusters larger than max_cluster_size
            large_bidx,large_kidx = (bin_size>max_cluster_size).nonzero(as_tuple=True)
            if len(large_bidx)>0:
                z = len(large_bidx)
                large_rows = mutual_choice[large_bidx,:,large_kidx] # z x n
                large_size = large_rows.sum(1).long() # z
                num_splits = torch.ceil(large_size / float(max_cluster_size)).long() # z
                max_num_splits = num_splits.max().item()
                largest_size = large_size.max().item()
                _,ones_idx = large_rows.topk(largest_size,dim=1,sorted=True) # z x n'
                spacing = ((large_size - max_cluster_size) / (num_splits - 1)).long() # z
                start_idx = torch.arange(max_num_splits,device=num_splits.device).long().unsqueeze(0).expand(z,-1) # z x mns
                start_idx_valid1, start_idx_valid2 = (start_idx<num_splits.unsqueeze(1)).nonzero(as_tuple=True) # tns
                start_idx = start_idx * spacing.unsqueeze(1)
                start_idx[torch.arange(z,device=start_idx.device).long(),num_splits-1] = large_size - max_cluster_size # in case uneven division
                start_batch_idx = torch.arange(z,device = start_idx.device).long().unsqueeze(1).expand(-1,max_num_splits)
                start_idx = start_idx[start_idx_valid1, start_idx_valid2] # tns, total number of splits
                start_batch_idx  = start_batch_idx[start_idx_valid1, start_idx_valid2]
                total_num_splits = len(start_idx_valid1) 
                assert total_num_splits == num_splits.sum(), "num splits incorrect!"
                fetch_idx = torch.arange(max_cluster_size,device=start_idx.device).long().unsqueeze(0).expand(total_num_splits,-1) # tns x mcs
                fetch_idx = fetch_idx + start_idx.unsqueeze(1)
                start_batch_idx = start_batch_idx.unsqueeze(1).expand(-1,max_cluster_size) # tns x mcs
                fetched_points = ones_idx[start_batch_idx.reshape(-1),fetch_idx.reshape(-1)].reshape(total_num_splits, max_cluster_size) # tns x mcs

                mutual_choice[large_bidx,:,large_kidx] = 0
                mutual_choice[large_bidx,:max_cluster_size,large_kidx] = 1
                member_idx = mutual_choice.permute(0,2,1).nonzero(as_tuple=True)[2].reshape(b,k,max_cluster_size) # b x k' x mcs

                fetched_points_z = torch.zeros(z, max_cluster_size, max_num_splits, device=fetched_points.device, dtype=fetched_points.dtype) 
                fetched_points_z[start_idx_valid1,:,start_idx_valid2] = fetched_points # z x mcs x mns
                member_idx[large_bidx,large_kidx] = fetched_points_z[:,:,0]

                second_idx = (start_idx_valid2>0).nonzero().view(-1)
                start_idx_valid1 = start_idx_valid1[second_idx]
                start_idx_valid2 = start_idx_valid2[second_idx] # remove the first splits
                fetched_points = fetched_points_z[start_idx_valid1,:,start_idx_valid2] # (tns-z) x mcs
                assert fetched_points.shape[0]==total_num_splits-z
                assert len(start_idx_valid1)==total_num_splits-z

                split_bidx = large_bidx[start_idx_valid1] # tns-z
                add_cluster_num = torch.bincount(split_bidx).max().long().item() # get the number of new clusters to add
                member_idx_expand = torch.zeros(b,k+add_cluster_num,max_cluster_size, device=member_idx.device, dtype=member_idx.dtype)
                member_idx_expand[:,:k] = member_idx 
                rotate_idx = torch.arange(add_cluster_num,device=member_idx.device).long().repeat(int(math.ceil((total_num_splits-z)/add_cluster_num)))[:(total_num_splits-z)] # tns - z
                member_idx_expand[split_bidx,rotate_idx+k] = fetched_points # b x k' x mcs
                member_idx = member_idx_expand 

                member_idx[inf_bidx,inf_kidx]=0
                valid_row_idx = (member_idx.sum(-1).reshape(-1) > 0).nonzero().view(-1) 
                if len(valid_row_idx)==b*(k+add_cluster_num):
                    valid_row_idx = None
            else:
                member_idx = mutual_choice.permute(0,2,1).nonzero(as_tuple=True)[2].reshape(b,k,max_cluster_size) # b x k x mcs
                if len(inf_bidx)==0:
                    valid_row_idx = None
                else:
                    member_idx[inf_bidx,inf_kidx]=0
                    valid_row_idx = (member_idx.sum(-1).reshape(-1) > 0).nonzero().view(-1) 
            return None, None, member_idx, valid_row_idx

    else:
        #start2 = time.time()
        dist = ((points[:,:,None,:] - means[:,None,:,:]) ** 2).sum(-1) # b x n x k
        if pos is not None:
            dist_pos = ((pos[:,:,None,:] - means_pos[:,None,:,:]) ** 2).sum(-1) # b x n x k
            dist += (pos_lambda / d * c) * dist_pos
        dist_orig = dist.clone()
        dist = dist / dist.sum(2,keepdim=True)
 
        '''
        mean_assignment0 = dist.argmin(dim=2,keepdim=True) # b x n x 1
        dist_overall0 = dist_orig.gather(dim=2,index=mean_assignment).mean()
        print("avg dist 0", dist_overall0, n)
        '''
 
        member_grab = dist.topk(n//k,dim=1,largest=False)[1] # b x n/k x k
        mutual_choice = torch.zeros(b,n,k,device=points.device)
        mutual_choice.scatter_(index=member_grab, dim=1, src=torch.ones(b,n//k,k,device=points.device))
        prop_num = mutual_choice.sum(2) # b x n
        overflow_idx = (prop_num.view(-1) > 1).nonzero().view(-1) # idx of points have >1 proposals
        if len(overflow_idx) > 0:
            chosen_prop_idx = mutual_choice.view(-1,k)[overflow_idx].multinomial(1).squeeze(1) # z
            chosen_prop = torch.zeros(len(overflow_idx),k,device=points.device)
            chosen_prop[torch.arange(len(overflow_idx),device=points.device),chosen_prop_idx] = 1
 
            mutual_choice.view(-1,k)[overflow_idx] = chosen_prop
            bidx, kidx = (mutual_choice.sum(1)<n//k).nonzero(as_tuple=True) # idx of too small cluster
            bidx2, nidx = (prop_num==0).nonzero(as_tuple=True)
            dist_inf = dist.new(dist.shape).zero_() + float('inf')
            dist_inf[bidx2,nidx]=dist[bidx2,nidx]
            dist = dist_inf
  
            ir=0
            max_round=11
            while True:
                ir += 1
                member_grab_num = int((n//k - mutual_choice.sum(1))[bidx,kidx].min().item())
                assert member_grab_num > 0, "member grab num shoud be > 0"
  
                member_grab = dist[bidx,:,kidx].topk(member_grab_num,dim=1,largest=False)[1].view(-1) # z*g
                mutual_choice[bidx.repeat_interleave(member_grab_num),member_grab,kidx.repeat_interleave(member_grab_num)] = 1
                prop_num = mutual_choice.sum(2) # b x n
                overflow_idx = (prop_num.view(-1) > 1).nonzero().view(-1) # idx of points have >1 proposals
                if len(overflow_idx)>0:
                    chosen_prop_idx = mutual_choice.view(-1,k)[overflow_idx].multinomial(1).squeeze(1) # z
                    chosen_prop = torch.zeros(len(overflow_idx),k,device=points.device)
                    chosen_prop[torch.arange(len(overflow_idx),device=points.device),chosen_prop_idx] = 1
                    mutual_choice.view(-1,k)[overflow_idx] = chosen_prop
                    del chosen_prop
                bidx, kidx = (mutual_choice.sum(1)<n//k).nonzero(as_tuple=True) # idx of too small cluster
                bidx2, nidx = (prop_num==0).nonzero(as_tuple=True)
                #print("num zero",len(nidx),ir)
                if len(kidx)==0:
                    assert len(nidx)==0, "nidx must have no members"
                    break
                if max_round is not None and ir==max_round:
                    num_repeats = (n//k-mutual_choice.sum(1))[bidx,kidx].long()
                    bidx = bidx.repeat_interleave(num_repeats)
                    kidx = kidx.repeat_interleave(num_repeats)
                    mutual_choice[bidx,nidx,kidx]=1
                    break
  
                dist_zero = dist.new(dist.shape).zero_()
                dist_zero[bidx,:,kidx] = dist[bidx,:,kidx]
                dist = dist_zero
                dist = dist / dist.sum(2,keepdim=True)
                dist_inf = dist.new(dist.shape).zero_() + float('inf')
                dist_inf[bidx2,nidx]=dist[bidx2,nidx]
                dist = dist_inf
 
        mean_assignment = mutual_choice.nonzero(as_tuple=True)[2].reshape(b,n,1) # b x n x 1
        #end2 = time.time()
        '''
        dist_overall = dist_orig.gather(dim=2,index=mean_assignment).mean()
        print("avg dist", dist_overall, n)
        '''

    bin_size = batched_bincount(mean_assignment.squeeze(2), valid_mask)
    max_bin_size = int(bin_size.max().item())
    '''
    print("top 3 bin size",bin_size.topk(3,dim=1)[0][:5],n)
    print("top 3 small bin size",bin_size.topk(3,dim=1,largest=False)[0][:5],n)
    '''
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
    end1 = time.time()
    #print("time perc", (end2-start2) / (end1-start1))
    
    return means, mean_assignment, reverse_assignment, valid_assignment_mask 



#if __name__ == '__main__':
