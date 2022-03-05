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
    img - B x c x H x W
    return point position and color - B x 2 x N, B x c x N
    '''
    b,c,h,w = img.shape
    pos = img.new(b,2,h,w).zero_().long()
    hs = torch.arange(0,h, device=img.device).long()
    ws = torch.arange(0,w, device=img.device).long()
    ys,xs = torch.meshgrid(hs,ws)
    xs=xs.unsqueeze(0).expand(b,-1,-1)
    ys=ys.unsqueeze(0).expand(b,-1,-1)
    pos[:,0]=xs
    pos[:,1]=ys
    pos = pos.view(b,2,-1)
    img = img.view(b,c,-1)
    return pos, img

def points2img(pos,pixel,h,w):
    '''
    convert points to img
    pos - B x 2 x N
    pixel - B x c x N
    h,w - scalar
    '''
    b,c,n = pixel.shape
    img = pos.new(b,c,h*w).zero_().to(pixel.dtype)
    idx = (pos[:,1]*w+pos[:,0]).long().unsqueeze(1).expand(-1,c,-1) # B x c x N
    img.scatter_(src=pixel, index=idx, dim=2)
    return img.view(b,c,h,w)

def cluster2points(cluster_pos, cluster_feat, cluster_mask, valid_row_idx, b, k, filter_invalid=False):
    '''
    cluster_pos - k' x m x d
    cluster_feat - k' x m x c
    cluster_mask - k' x m x 1, can be None
    valid_row_idx - can be None
    b - batch number
    k - cluster number
    '''
    _, m, c = cluster_feat.shape
    d = cluster_pos.shape[2]
    irregular = cluster_mask is not None
    if valid_row_idx is not None:
        assert irregular, "cluster mask must not be None"
        new_pos = cluster_pos.new(b*k,m,d).zero_().long()
        new_feat = cluster_feat.new(b*k,m,c).zero_()
        new_mask = cluster_mask.new(b*k,m,1).zero_().long()
        new_feat[valid_row_idx] = cluster_feat
        new_pos[valid_row_idx] = cluster_pos
        new_mask[valid_row_idx] = cluster_mask
    else:
        new_feat = cluster_feat
        new_pos = cluster_pos
        new_mask = cluster_mask 

    new_feat = new_feat.reshape(b,k,m,c).permute(0,3,1,2).reshape(b,c,-1) # b x c x n
    new_pos = new_pos.reshape(b,k,m,d).permute(0,3,1,2).reshape(b,d,-1) # b x d x n
    if irregular:
        new_mask = new_mask.reshape(b,k,m,1).permute(0,3,1,2).reshape(b,1,-1) # b x 1 x n

    if irregular and filter_invalid:
        largest_n = new_mask.sum(2).max() # largest sample size
        valid_idx = new_mask.view(-1).nonzero().squeeze() # z
        batch_idx = torch.arange(b,device=valid_idx.device).long().unsqueeze(1).expand(-1,k*m).reshape(-1)[valid_idx] # z

        valid_feat = new_feat.permute(0,2,1).view(-1,c)[valid_idx] # z x c
        valid_pos = new_pos.permute(0,2,1).view(-1,d)[valid_idx] # z x d
        valid_mask = new_mask.permute(0,2,1).view(-1,1)[valid_idx] # z x 1
        z = len(valid_idx)
        rotate_idx = torch.arange(largest_n,device=valid_mask.device).long().repeat(torch.ceil(z/largest_n).long().item())[:z]
        new_pos = cluster_pos.new(b,d,largest_n).zero_().long()
        new_feat = cluster_feat.new(b,c,largest_n).zero_()
        new_mask = cluster_feat.new(b,1,largest_n).zero_().long()
        new_pos[batch_idx,:,rotate_idx] = valid_pos
        new_feat[batch_idx,:,rotate_idx] = valid_feat
        new_mask[batch_idx,:,rotate_idx] = valid_mask
        if new_mask.min() == 1:
            new_mask = None
    return new_pos, new_feat, new_mask

def points2cluster(pos, feat, member_idx, cluster_mask):
    '''
    pos - b x d x n
    feat - b x c x n
    member_idx - b x m x k
    cluster_mask - b x m x k
    '''
    b,m,k = member_idx.shape
    _,c,n = feat.shape
    d = pos.shape[1]
    batch_tmp = torch.arange(b,device=feat.device).long().unsqueeze(1).expand(-1,m*k)
    member_idx = member_idx.view(b,-1)
    member_idx = (batch_tmp*n+member_idx).view(-1)
    cluster_pos = pos.permute(0,2,1).reshape(-1,d)[member_idx].reshape(b,m,k,d).permute(0,2,1,3).reshape(-1,m,d)
    cluster_feat = feat.permute(0,2,1).reshape(-1,c)[member_idx].reshape(b,m,k,c).permute(0,2,1,3).reshape(-1,m,c)
    # get valid cluster id
    irregular = cluster_mask is not None and cluster_mask.min() == 0
    if irregular:
        valid_row = (cluster_mask.sum(1) > 0).long() # b x k
        valid_row_idx = valid_row.view(-1).nonzero().squeeze() # z
        cluster_mask = cluster_mask.permute(0,2,1).reshape(-1,m).unsqueeze(2) # k' x m x 1
        if len(valid_row_idx) == b*k:
            valid_row_idx = None
        else:
            cluster_pos = cluster_pos[valid_row_idx]
            cluster_feat = cluster_feat[valid_row_idx]
            cluster_mask = cluster_mask[valid_row_idx]
        cluster_pos *= cluster_mask
        cluster_feat *= cluster_mask
    else:
        cluster_mask = None
        valid_row_idx = None
    return cluster_pos, cluster_feat, cluster_mask, valid_row_idx

def random_sampling(pc_pos, feature, num_samples):
    '''
    randomly sample points
    pc_pos - B x 3 x N
    feature - B x c x N
    num_samples - scalar
    return sample_idx - B x n
    '''
    b,c,n=feature.size()
    sample_idx = pc_pos.new(b,num_samples).zero_()
    torch.manual_seed(0)
    '''
    for i in range(b):
        sample_idx[i]=torch.randperm(n)[:num_samples]
        '''
    sample_idx_ = (torch.randperm(n)[:num_samples]).repeat(b,1) # b x n
    sample_idx[:]=sample_idx_[:]
    sample_idx = sample_idx.unsqueeze(1).long()  # B x 1 x n
    sampled_pos = pc_pos.gather(dim=2, index=sample_idx.expand(-1,3,-1)) # B x 3 x n
    sampled_feature = feature.gather(dim=2, index=sample_idx.expand(-1,c,-1)) # B x c x n
    return sampled_pos, sampled_feature, sample_idx

def attention_sampling(pc_pos, feature, attention, num_samples):
    '''
    sample points according to attention
    pc_pos - B x 3 x N
    feature - B x c x N
    attention - B x 1 x N
    num_samples - scalar
    return sample_idx - B x n
    '''
    b,c,n = feature.size()
    sample_idx = torch.multinomial(attention.squeeze(1), num_samples) # B x n
    sample_idx = sample_idx.unsqueeze(1)  # B x 1 x n
    sampled_pos = pc_pos.gather(dim=2, index=sample_idx.expand(-1,3,-1)) # B x 3 x n
    sampled_feature = feature.gather(dim=2, index=sample_idx.expand(-1,c,-1)) # B x c x n
    sampled_atten = attention.gather(dim=2, index=sample_idx) # B x 1 x n
    return sampled_pos, sampled_feature, sampled_atten, sample_idx

def inverse_density_sampling(inputs, num_samples, k = 20):
    """
    Inverse density sampling
    inputs - B x 3 x N
    num_samples - number of sampled points
    k - number of neighbors when estimating density

    return indices of sampled points - B x n
    """
    n = inputs.size(2)
    #(b, n, n)
    pair_dist = pairwise_distance(inputs)
    if k > n:
        k = n
    pair_dist=pair_dist.contiguous()
    distances, _ = pair_dist.topk(dim=1,k=k,largest=False) # B x K x N

    #(b, n)
    distances_avg = torch.abs(torch.mean(distances, dim=1)) + 1e-8 # B x N
    prob_matrix = distances_avg / torch.sum(distances_avg, dim = 1, keepdim=True) # B x N

    #(b, num_samples)
    sample_idx = torch.multinomial(prob_matrix, num_samples) # B x n
    return sample_idx

def kernel_density_estimation(nn_pos,sigma,normalize = False):
    '''
    Calculate the kernel density estimation using Gaussian kernal and k nearest neighbors of the N points
    nn_pos - B x 3 x K x N, the xyz position of the K neighbors of all the N points relative to the center
    sigma - the bandwidth of the Gaussian kernel
    normalize - normalize it using the largest density among the N points

    return density - B x 1 x N
    '''
    sigma_=nn_pos.new(1).float()
    sigma_[0]=sigma
    sigma=sigma_.clone()
    posdivsig = nn_pos.div(sigma) # x/sig, y/sig, z/sig
    quadform = posdivsig.pow(2).sum(dim=1) # (x^2+y^2+z^2)/sig^2
    #print(quadform) # should be B x K x N
    logsqrtdetSigma = sigma.log() * 3 # log(sigma^3)
    twopi=sigma.clone()
    twopi[0]=2 * 3.1415926
    mvnpdf = torch.exp(-0.5 * quadform - logsqrtdetSigma - 1.5 * torch.log(twopi)) #(2pi)^(-3/2)*sigma^(-3)*exp(-0.5*(x^2+y^2+z^2)/sig^2)
    mvnpdf = torch.sum(mvnpdf, dim = 1, keepdim = True) # sum all neighbors
    #print(mvnpdf) # should be B x 1 x N

    scale = 1.0 / nn_pos.shape[2] #1/K
    density = mvnpdf*scale # B x 1 x N

    if normalize:
        density_max,_ = density.max(dim=2, keepdim=True) # B x 1 x 1
        density = density.div(density_max)

    return density

def pairwise_distance(input_pos):
    """
    Args:
    input_pos: tensor(batch_size, num_dims, num_points)
    Returns:
    pairwise distance: (batch_size, num_points, num_points)
    """
    b = input_pos.size(0)
    input_pos_transpose = input_pos.contiguous().permute(0, 2, 1)
    input_pos_inner = torch.matmul(input_pos_transpose, input_pos)
    input_pos_inner = -2 * input_pos_inner
    input_pos_square = torch.sum(input_pos * input_pos, dim = 1, keepdim=True)
    input_pos_square_transpose = input_pos_square.contiguous().permute(0, 2, 1)
    return input_pos_inner + input_pos_square + input_pos_square_transpose

def pairwise_distance_general(queries, input_pos):
    '''
    Args:
    queries: (batch_size, num_dims, num_points')
    input_pos: tensor(batch_size, num_dims, num_points)
    Returns:
    pairwise distance: (batch_size, num_points, num_points')
    '''
    #(b, n, c)
    input_pos_transpose = input_pos.contiguous().permute(0, 2, 1)
    #(b, n, n')
    inner = torch.matmul(input_pos_transpose, queries)
    inner = -2 * inner
    #(b, n, 1)
    input_pos_square_transpose = torch.sum(input_pos_transpose * input_pos_transpose, dim = 2, keepdim=True)
    #(b, 1, n')
    queries_square =  torch.sum(queries * queries, dim = 1, keepdim=True)
    return queries_square + inner + input_pos_square_transpose

def knn(dist, k=20, ret_dist=False):
    """
    Get KNN based on dist matrix
    Args:
    dist: (batch_size, num_points, num_points)
    k:int
    Returns:
    nearest neighbors: (batch_size, k, num_points)
    """
    dist = dist.contiguous()
    n = dist.size(1)
    if k > n:
        k = n
    top_dist, nn_idx = dist.topk(k=k,dim=1,largest=False)
    if ret_dist:
        return top_dist,nn_idx
    else:
        return nn_idx

def knn_keops(query, database, k, return_dist = False):
    '''
    get knn using pykeops library
    query - b x c x n
    database - b x c x N
    k - scalar
    return nn_dix - b x k x n
    '''
    db_orig = database
    q_orig = query
    from pykeops.torch import LazyTensor
    query = query.permute(0,2,1).contiguous()
    database = database.permute(0,2,1).contiguous()
    query_ = LazyTensor(query[:,None,:,:])
    database_ = LazyTensor(database[:,:,None,:])
    dist = ((query_-database_) ** 2).sum(-1) # b x N x n
    nn_idx = dist.argKmin(k, dim=1) # b x n x k
    nn_idx = nn_idx.permute(0,2,1).contiguous()
    if return_dist:
        nn_pos = gather_nd(db_orig, nn_idx) # b x 3 x k x n
        nn_dist = ((nn_pos-q_orig.unsqueeze(2))**2).sum(1) # b x k x n
        '''
        #nn_dist = dist.gather(dim=1,index=nn_idx) # b x k x n
        nn_dist = dist.Kmin(k, dim=1) # b x n x k
        nn_dist = nn_dist.permute(0,2,1).contiguous()
        '''
        return nn_idx, nn_dist
    return nn_idx

def epsilon_ball(query, databse, radius, k, return_dist = False):
    '''
    get knn within ball of radius
    query - b x c x n
    database - b x c x N
    k - at most k neighbors
    radius - scalar, radius of the ball
    return nn_dix - b x k x n
    nn_dist - b x k x n
    mask - 1 means inside ball, b x k x n
    '''
    nn_idx, nn_dist = knn_keops(query, database, k, return_dist = True) # b x k x n, b x k x n
    mask = (nn_dist<=radius.pow(2)).long() # b x k x n
    max_nei = mask.sum(1).max()
    if max_nei < k:
        nn_idx = nn_idx[:,:max_nei]
        nn_dist = nn_dist[:,:max_nei]
        mask = mask[:,:max_nei]
    if return_dist:
        return nn_idx, nn_dist, mask
    return nn_idx, mask

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
    result = torch.zeros(b,k,device = mat.device).long()
    ones = torch.ones(mat.shape,device = mat.device).long()
    if valid_mask is not None:
        ones *= valid_mask
    result.scatter_add_(dim=1, index=mat, src=ones) # b x k
    return result

def init_kmeanspp(points, k):
    '''
    initialize kmeans++
    points - b x n x c
    return
    means - b x k x c
    '''
    from pykeops.torch import LazyTensor
    # get a random point
    b,n,c = points.shape
    idx = torch.randint(n,(b,)) # b
    centers = points[torch.arange(b),idx] # b x c
    centers = centers.unsqueeze(1) # b x 1 x c
    for _ in range(k-1):
        points_ = LazyTensor(points[:,:,None,:]) # b x n x 1 x c
        centers_ = LazyTensor(centers[:,None,:,:]) # b x 1 x k x c
        dist = ((points_ - centers_) ** 2).sum(-1) # b x n x k
        dist_min = dist.min(dim=2)
        dist_min = dist_min.squeeze(2) # b x n
        idx = (dist_min+1e-12).multinomial(num_samples=1).squeeze(1) # b x 1
        new_center = points[torch.arange(b),idx] # b x c
        new_center = new_center.unsqueeze(1) # b x 1 x c
        centers = torch.cat([centers, new_center], dim=1)
    return centers

def kmeans_keops(points, k, max_cluster_size=None, num_nearest_mean=1, num_iter=10, pos = None, pos_lambda=1, valid_mask=None, init='random', init_feat_means=None, init_pos_means=None, normalize=True):
    '''
    points - b x c x n
    k - number of means
    pos - postion of points, b x d x n
    pos_lambda - lambda of pos in dist calculation
    valid_mask - b x 1 x n, binary mask indicating the valid points
    init - method of initialization, kmeans++ or random
    init_feat_means - initialize using these means, b x k x c 
    max_cluster_size - do random sampling in larger clusters; must be >= n/k
                 only affects reverse_assignment and valid_assignment_mask
    normalize - whether to normalize points to mean 0 std 1
    return
    means - b x c x k
    mean_assignment - b x num_nearest_mean x n
    reverse_assignment - b x m x k, m is the largest cluster size, invalid position filled with 0
    valid_assignment_mask - b x m x k, if sum along m gets 0, then the cluster is invalid
    '''
    #max_cluster_size=25
    points = points.detach()
    if pos is not None:
        pos = pos.detach()
    old_dtype = points.dtype
    points = points.to(torch.float32)
    if valid_mask is not None:
        valid_mask = valid_mask.detach().long()
        points *= valid_mask
        if pos is not None:
            pos *= valid_mask # make sure invalid pos and points are all 0
        valid_mask = valid_mask.squeeze(1) # b x n
        num_valid = valid_mask.sum() # total number of valid points
    from pykeops.torch import LazyTensor
    points = points.permute(0,2,1).contiguous() # b x n x c
    b,n,c = points.shape
    if max_cluster_size is not None:
        assert max_cluster_size>= math.ceil(n/k), "max_cluster_size should not be smaller than average"
    if pos is not None:
        d = pos.shape[1]
        pos = pos.to(points.dtype)
        pos = pos.permute(0,2,1).contiguous() # b x n x d
        # normalize mean and std
    if normalize:
        feat_mean = points.mean()
        feat_std = (points.view(-1).var(dim=0, unbiased=False)+1e-5).pow(0.5)
        points = (points-feat_mean) / feat_std
        if pos is not None:
            pos_mean = pos.mean()
            pos_std = (pos.view(-1).var(dim=0, unbiased=False)+1e-5).pow(0.5)
            pos = (pos-pos_mean) / pos_std
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
    elif init=='kmeans++': # only can run on a single gpu
        means = init_kmeanspp(points, k) # b x k x c
        if pos is not None:
            means_pos = init_kmeanspp(pos, k) # b x k x d
    points_ = LazyTensor(points[:,:,None,:]) # b x n x 1 x c
    means_ = LazyTensor(means[:,None,:,:]) # b x 1 x k x c
    if pos is not None:
        pos_ = LazyTensor(pos[:,:,None,:]) # b x n x 1 x d
        means_pos_ = LazyTensor(means_pos[:,None,:,:]) # b x 1 x k x d

    if init=='random' and init_feat_means is None and valid_mask is not None:
        # turn excessive invalid means to nan
        means_valid_mask = valid_mask[:,rand_idx].unsqueeze(2) # b x k x 1
        row_sum = means_valid_mask.sum(1)[:,0] # b, check if all are invalid
        all_zero_index = (row_sum==0).long().nonzero().squeeze()
        means_valid_mask[all_zero_index, 0] = 1
        nan_mask = means_valid_mask / means_valid_mask # 1 is 1, 0 becomes nan
        means = means * nan_mask
        means_ = LazyTensor(means[:,None,:,:]) # b x 1 x k x c
        if pos is not None:
            means_pos = means_pos * nan_mask
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

    max_bin_size = batched_bincount(mean_assignment.squeeze(2), valid_mask).max().item()
    #print("max bin size",max_bin_size, "avg size", n//k)
    if max_cluster_size is not None:
        max_bin_size = min(max_cluster_size, max_bin_size)
    # get reverse_assignment
    sorted_assignment, sorted_point_idx = mean_assignment.squeeze(2).sort(dim=-1, descending=False) # b x n, b x n
    if valid_mask is not None:
        sorted_valid_mask = valid_mask.gather(index=sorted_point_idx,dim=-1) # b x n
        sorted_valid_mask = sorted_valid_mask.reshape(-1)
    sorted_assignment = sorted_assignment.reshape(-1)
    sorted_point_idx = sorted_point_idx.reshape(-1)
    if valid_mask is not None:
        sorted_valid_idx = sorted_valid_mask.nonzero().squeeze()
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

    final_idx = (batch_idx, rotate_idx, sorted_assignment)
    reverse_assignment = torch.zeros(b, max_bin_size, k, device = mean_assignment.device).long()
    reverse_assignment -= 1 # blank space filled with -1
    reverse_assignment.index_put_(indices=final_idx, values=sorted_point_idx) # b x m x k
    valid_assignment_mask = (reverse_assignment > -1).long() # b x m x k
    reverse_assignment.clamp_(min=0)
    
    if num_nearest_mean > 1:
        mean_assignment = dist.argKmin(num_nearest_mean,dim=2).long() # b x n x num_mean

    means = means.to(old_dtype)
    del points
    if pos is not None:
        del pos

    return means.permute(0,2,1), mean_assignment.permute(0,2,1), reverse_assignment, valid_assignment_mask 


def gather_nd(inputs, nn_idx):
    """
    input: (batch_size, num_dim, num_points)
    nn_idx:(batch_size, k, num_points)
    output:
    output:(batch_size, num_dim, k, num_points)
    """
    b, c, _ = inputs.size()
    _, k, n = nn_idx.size()

    # (b, c, k*n)
    nn_idx = nn_idx.unsqueeze(dim=1).expand(-1, c, -1, -1).view(b, c, -1)
    inputs_gather = inputs.gather(dim=-1, index=nn_idx)
    inputs_gather = inputs_gather.view(b, c, k, n)
    return inputs_gather

def get_inverse_density(pc_pos,k,sigma):
    # get density of every point
    pairwise_dist= pairwise_distance(pc_pos) # B x N x N
    nn_idx= knn(pairwise_dist, k)
    nn_pos= gather_nd(pc_pos, nn_idx)
    lnn_pos= nn_pos-pc_pos.unsqueeze(dim=2)
    density = kernel_density_estimation(lnn_pos,sigma,False) # density of all input points - B x 1 x N
    one = pc_pos.new(1)
    one[0]=1.0
    inverse_density = one.div(density)
    return inverse_density

def gaussian_decay_weights(dist,sigma,tv_norm=False):
    '''
    dist - B x k x N
    return
    weights - B x k x N
    '''
    sigma_=dist.new(1).float()
    sigma_[0]=sigma
    sigma=sigma_.clone()
    weights=dist.div(sigma).pow(2).mul(-0.5).exp()
    if tv_norm:
        w_sum = weights.sum(1).mean(1)
    else:
        w_sum = weights.sum(dim=1,keepdim=True)
    weights/=w_sum
    return weights

def shepard_decay_weights(dist,power=3):
    '''
    dist - B x k x N
    return
    weights - B x k x N
    decay weights using inverse powered dist
    '''
    ipd = 1.0/(dist.pow(power)+10e-12)
    weights = ipd / ipd.sum(dim=1,keepdim=True)
    return weights
    
def upsample_feature_shepard(pc_pos,sampled_pos,feature,sampled_idx=None,k=10):
    '''
    upsample the feature with fewer points than pc_pos using shepard method
    pc_pos - B x 3 x N
    sampled_pos - pos of sampled points, B x 3 x n
    feature - B x c x n
    sampled_idx - idx of the sampled points, B x 1 x n
    k - number of points in neighborhood
    return feature' - B x c x N
    '''
    b,c,n = feature.shape
    if k>n:
        k=n
    nn_idx, nn_dist = knn_keops(pc_pos, sampled_pos, k=k, return_dist = True)
    #pairwise_dist = pairwise_distance_general(pc_pos, sampled_pos) # B x n x N
    #nn_dist,nn_idx = knn(pairwise_dist, k=k, ret_dist=True) # nearest sample dist and index - B x K x N, B x K x N

    nn_weights=shepard_decay_weights(nn_dist) # B x K x N, weights of the samples
    nn_weights_fat = nn_weights.unsqueeze(1).expand(-1,c,-1,-1)
    nn_features = gather_nd(feature, nn_idx) # B x c x K x N, features of the neighbor samples
    up_features = nn_features.mul(nn_weights_fat).sum(dim=2) # B x c x N

    if sampled_idx is not None:
        up_features.scatter_(dim=2, index=sampled_idx.long().expand(-1,c,-1), src=feature) # keep feature at sampled point unchanged

    return up_features

def upsample_gradient(pc_pos, sampled_pos, feature, sampled_idx=None, k=10):
    '''
    pc_pos - B x 3 x N
    sampled_pos - pos of sampled points, B x 3 x n
    feature - B x c x n
    sampled_idx - idx of the sampled points, B x 1 x n
    k - number of points in neighborhood
    return feature' - B x c x N
    '''
    b,c,n = feature.shape
    if k>n:
        k=n
    nn_idx, nn_dist = knn_keops(pc_pos, sampled_pos, k=k, return_dist = True)
    nn_weights=shepard_decay_weights(nn_dist) # B x K x N, weights of the samples
    nn_weights = nn_weights[:,1:]
    nn_features = gather_nd(feature, nn_idx) # B x c x K x N, features of the neighbor samples
    nn_pos = gather_nd(sampled_pos, nn_idx) # B x 3 x K x N, pos of the neighbor samples

    x0 = nn_pos[:,:,0:1,:].clone() # randomly choose one as x0, B x 3 x 1 x N
    x0_feat = nn_features[:,:,0:1,:].clone() # randomly choose one as x0, B x c x 1 x N
    pos_diff = (nn_pos-x0)[:,:,1:] # B x 3 x (K-1) x N
    pos_diff_2 = pc_pos - x0.squeeze(2) # B x 3 x N
    feat_diff = (nn_features-x0_feat)[:,:,1:] # B x c x (K-1) x N

    diff_prod = (pos_diff_2.unsqueeze(2) * pos_diff).sum(1) # b x (k-1) x N
    pos_diff_len = pos_diff.pow(2).sum(1) # b x (k-1) x N
    pos_diff2_len = pos_diff_2.pow(2).sum(1,keepdim=True) # b x 1 x N
    ratio = pos_diff2_len / (diff_prod+10e-12) # b x (k-1) x N
    ratio.clamp_(-2.0,2.0)
    angle = diff_prod / (pos_diff_len.pow(0.5)*pos_diff2_len.pow(0.5)+10e-12) # b x (k-1) x N
    angle_weights = angle.abs().pow(7) # the more perpendicular, the smaller the weight
    nn_weights = nn_weights / nn_weights.sum(1,keepdim=True)
    angle_weights = angle_weights / angle_weights.sum(1,keepdim=True)
    total_weights = nn_weights * angle_weights
    total_weights = total_weights / total_weights.sum(1,keepdim=True)

    up_feat = ratio.unsqueeze(1) * feat_diff + x0_feat # b x c x (k-1) x N
    up_feat = (up_feat * total_weights.unsqueeze(1)).sum(2) # b x c x n

    if sampled_idx is not None:
        up_feat.scatter_(dim=2, index=sampled_idx.long().expand(-1,c,-1), src=feature) # keep feature at sampled point unchanged
    
    return up_feat
    


def upsample_feature_nearest(pc_pos,sampled_pos,feature):
    '''
    upsample the feature with fewer points than pc_pos using nearest neighbor
    '''
    b,c,n = feature.shape
    start = time.time()
    pairwise_dist = pairwise_distance_general(pc_pos, sampled_pos) # B x n x N
    pw_time = time.time()
    print("pairwise time:",pw_time-start)
    nn_idx = knn(pairwise_dist, k=1, ret_dist=False) # nearest sample index - B x 1 x N
    knn_time = time.time()
    print("knn time:",knn_time-pw_time)
    nn_features = gather_nd(feature, nn_idx).squeeze(2) # B x c x N, features of the neighbor samples
    nn_feat_time = time.time()
    print("nn_feat time:",nn_feat_time-knn_time)
    return nn_features

def build_weight_filter(radius,power=3):
    hs = torch.arange(0,radius).float()
    ws = torch.arange(0,radius).float()
    ys,xs = torch.meshgrid(hs,ws)
    dis = (xs.pow(2)+ys.pow(2)).pow(0.5)
    dis[0,0]=10e-8
    rb_weights = 1.0/dis.pow(power) # right bottom weights
    lb_weights = rb_weights.flip(1) 
    b_weights = torch.cat([lb_weights[:,:-1],rb_weights],dim=1)
    t_weights = b_weights.flip(0)
    weights = torch.cat([t_weights[:-1],b_weights],dim=0)
    weights = weights.unsqueeze(0).unsqueeze(1) # unnormalized, 1 x 1 x r x r
    return weights

def upsample_feature_conv(sampled_pos,feature,h,w,filter_radius):
    '''
    only apply to 1 channel img point clouds
    '''
    b,c,n = feature.shape
    img = points2img(sampled_pos,feature,h,w) # b x 1 x h x w
    weight_filter = img.new(1,1,2*filter_radius-1,2*filter_radius-1)
    weight_filter[:] = build_weight_filter(filter_radius)
    sample_filter = (img!=0).float()
    weight_sum = F.conv2d(sample_filter, weight_filter, padding = filter_radius-1) # sum of weights in neighborhood
    feature_sum = F.conv2d(img, weight_filter, padding = filter_radius-1) # weighed sum of feature in neighborhood
    up_feature = feature_sum / weight_sum
    up_feature = (1-sample_filter)*up_feature+sample_filter*img
    _, pixel = img2points(up_feature)
    #print((nn.AvgPool2d(2*filter_radius-1,stride = 1,padding = filter_radius-1)(sample_filter)*(2*filter_radius-1)*(2*filter_radius-1)).mean()) # avg number of valid neighbor
    return pixel

if __name__ == '__main__':
    # test the upsampling method
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    h=40
    w=50
    n=5
    img = np.zeros((h,w))
    '''
    np.random.seed(0)
    xs = np.random.permutation(w)[:n]
    ys = np.random.permutation(h)[:n]
    zs = np.random.permutation(100)[:n]
    '''
    xs = np.asarray([0,49,0,49,25])
    ys = np.asarray([0,0,39,39,20])
    zs = np.asarray([0,0,0,0,100])
    print(xs,ys,zs)
    for i in range(len(xs)):
        img[ys[i],xs[i]]=zs[i]

    sampled_pos = np.zeros((3,n))
    sampled_pos[0]=xs
    sampled_pos[1]=ys
    sampled_pos = torch.Tensor(sampled_pos).unsqueeze(0)
    feature = torch.Tensor(zs).unsqueeze(0).unsqueeze(0)
    sampled_idx = torch.Tensor(ys*w+xs).unsqueeze(0).unsqueeze(0)

    img = torch.Tensor(img).unsqueeze(0).unsqueeze(0)
    pos,pixel = img2points(img) # 1 x 1 x N

    pixel2 = upsample_feature(pos, sampled_pos, feature, sampled_idx,k=10)

    sampled_pos2, sampled_feature2, _, _, = attention_sampling(pos, pixel2, pixel2, n)
    pixel3 = points2img(sampled_pos2, sampled_feature2, h,w)

    pos = pos.squeeze().view(3,h,w).numpy()
    pixel = pixel.squeeze().view(h,w).numpy()
    pixel2 = pixel2.squeeze().view(h,w).numpy()
    pixel3 = pixel3.squeeze().view(h,w).numpy()

    fig, axs = plt.subplots(3,subplot_kw={'projection': '3d'})

    x_data = pos[0].ravel()
    y_data = pos[1].ravel()
    pixel = pixel.ravel()
    pixel2 = pixel2.ravel()
    pixel3 = pixel3.ravel()

    axs[0].bar3d(x_data,y_data, 0,1,1,pixel,zsort='average')
    axs[1].bar3d(x_data,y_data, 0,1,1,pixel2,zsort='average')
    axs[2].bar3d(x_data,y_data, 0,1,1,pixel3,zsort='average')
    plt.show()

