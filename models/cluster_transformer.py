# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .point_utils import points2img, kmeans, cluster2points, points2cluster, batched_bincount
import torch_scatter

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ClusterAttention(nn.Module):
    """
    Performs cluster attention on points after k-means

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        pos_dim: dimension of x,y coordinates
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        pos_mlp_bias: add learnable bias to pos mlp
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, pos_dim=2, qkv_bias=True, pos_mlp_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., output_dim=None):

        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.pos_mlp = nn.Linear(pos_dim, num_heads, bias=pos_mlp_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if output_dim is None:
            self.proj = nn.Linear(dim, dim)
        else:
            self.proj = nn.Linear(dim, output_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, pos, feat, cluster_feat, cluster_score, mean_assignment, mask, member_idx, batch_idx, k, valid_row_idx, attend_means, cluster_mask=None, q_subsample_idx=None):
        """
        Args:
            pos - b x n x d 
            feat - b x n x c 
            mask - b x n x 1
            cluster_feat - b x n x c_, feat used in kmeans clustering
            cluster_score - z x m, contribution to cluster means, when in attn-to-means
            mean_assignment - b x n x num_clus, index of nearest centroids
            member_idx, z x m
            batch_idx, z x m
            valid_row_idx, z
            cluster_mask, z x m
            k, number of clusters per sample
            q_subsample_idx, b x n' x 1
        """
        cluster_size = 4
        if mean_assignment is None:
            nnc = 1
        else:
            nnc = mean_assignment.shape[-1] # number of nearest clusters

        b,n,c=feat.shape
        d = pos.shape[2]
        assert c == self.dim, "dim does not accord to input"
        assert d == self.pos_dim, "pos dim does not accord to input"

        h = self.num_heads
        c_ = c // h
        qkv = self.qkv(feat) # b x n x (3*c)

        qkv = qkv.reshape(b,n,3,c)
        q = qkv[:,:,0]
        kv = qkv[:,:,1:].reshape(b,n,-1)
        pos_orig = pos.clone() # b x n x d
        cluster_feat_orig = cluster_feat.clone() # b x n x c_

        if q_subsample_idx is not None:
            q = q.gather(index=q_subsample_idx.expand(-1,-1,c),dim=1) # b x n' x c
            mean_assignment = mean_assignment.gather(index=q_subsample_idx.expand(-1,-1,nnc),dim=1) # b x n' x nnc
            pos_orig = pos_orig.gather(index=q_subsample_idx.expand(-1,-1,d),dim=1) # b x n' x d
            cluster_feat_orig = cluster_feat_orig.gather(index=q_subsample_idx.expand(-1,-1,c_),dim=1) # b x n' x c_
            if mask is not None:
                mask_orig = mask.gather(index=q_subsample_idx,dim=1) # b x n' x 1
            else:
                mask_orig = None
            n = q.shape[1]

        '''
        '''
        pos = pos.to(feat.dtype)
        pos = pos / pos.view(-1,d).max(0)[0] # normalize
        
        if member_idx is not None:
            z,m = member_idx.shape

            if not attend_means:
                # collect nearest clusters for each point
                batch_idx2 = torch.arange(b,device=mean_assignment.device,dtype=mean_assignment.dtype).reshape(-1,1,1).expand(-1,n,nnc) # b x n x nnc
                member_idx = member_idx.reshape(b,k,m)[batch_idx2.reshape(-1),mean_assignment.reshape(-1)].reshape(b,n,nnc*m) # b x n x nnc*m
                self_idx_rotate = torch.arange(n,device=feat.device).long().repeat(b).unsqueeze(-1)
                if q_subsample_idx is not None:
                    self_idx_rotate = q_subsample_idx.reshape(b*n,1)
                self_idx = (member_idx.reshape(b*n,-1)==self_idx_rotate).nonzero(as_tuple=True)
                if cluster_mask is not None:
                    cluster_mask = cluster_mask.reshape(b,k,m)[batch_idx2.reshape(-1),mean_assignment.reshape(-1)].reshape(b*n,nnc*m) # b*n x nnc*m
                    cluster_mask[self_idx]=0
                    # sample a subset of neighbors
                    neighbor_idx = cluster_mask.to(feat.dtype).clamp(min=1e-5).multinomial(cluster_size-1).reshape(b,n,cluster_size-1) # b x n x cluster_size-1
                    cluster_mask = cluster_mask.reshape(b,n,-1)
                else:
                    ones = torch.ones_like(member_idx).reshape(b*n,-1)
                    ones[self_idx]=1e-5
                    neighbor_idx = ones.multinomial(cluster_size-1).reshape(b,n,cluster_size-1) # b x n x cluster_size-1
                member_idx = member_idx.gather(index=neighbor_idx,dim=-1) # b x n x m-1
                member_idx = torch.cat([member_idx,self_idx_rotate.reshape(b,n,1)],dim=-1) # b x n x cluster_size
                if cluster_mask is not None:
                    cluster_mask = cluster_mask.gather(index=neighbor_idx,dim=-1) # b x n x m-1
                    cluster_mask = torch.cat([cluster_mask,torch.ones(b,n,1,device=cluster_mask.device,dtype=cluster_mask.dtype)],dim=-1) # b x n x cluster_size
                batch_idx = torch.arange(b,device=mean_assignment.device,dtype=mean_assignment.dtype).reshape(-1,1,1).expand(-1,n,cluster_size) # b x n x m
                m = cluster_size
                z = b*n

            member_idx = member_idx.reshape(-1) # z*m
            batch_idx = batch_idx.reshape(-1) # z*m
            kv = kv[batch_idx,member_idx].clone().reshape(z,m,-1)
            pos = pos[batch_idx,member_idx].clone().reshape(z,m,d)
            if not attend_means:
                cluster_feat = cluster_feat[batch_idx,member_idx].clone().reshape(z,m,c_)
            if cluster_mask is not None:
                mask = cluster_mask.reshape(z,m,1)
            elif mask is not None:
                mask = mask[batch_idx,member_idx].clone().reshape(z,m,1)
                if mask.min()==1:
                    mask = None
        else:
            z,m=b,n
            if not attend_means:
                z = b*n
                m = cluster_size

        if attend_means:
            q = q.reshape(b,n,h,c_).permute(0,2,1,3) # b x h x n x c_
            if cluster_score is None:
                cluster_score = torch.ones(z,m,1,device=kv.device)
                if mask is not None:
                    cluster_score = cluster_score / (mask.sum(1,keepdim=True)+1e-5) # z x m x 1
                    cluster_score = cluster_score * mask
                else:
                    cluster_score = cluster_score / m
            else:
                if mask is not None:
                    cluster_score = cluster_score.unsqueeze(-1) * mask
            kv = (kv * cluster_score).sum(1).reshape(b,k,2,h,c_) # b x k x 2 x h x c_
            pos = (pos * cluster_score).sum(1).reshape(b,k,d) # b x k x d
            kv = kv.permute(2,0,3,1,4) # 2 x b x h x k x c_
            key,v = kv[0],kv[1] # b x h x k x c_
            if mask is not None:
                mask = (mask.sum(1)>0).long().reshape(b,1,1,k) # b x 1 x 1 x k
                if mask.min()==1:
                    mask = None

        else:
            kv = kv.reshape(b,n,m,2,h,c_).permute(3,0,4,1,2,5) # 2 x b x h x n x m x c_
            key, v = kv[0], kv[1]  # b x h x n x m x c_
            q = q.reshape(b,n,1,h,c_).permute(0,3,1,2,4) # b x h x n x 1 x c_
            if mask is not None:
                mask = mask.reshape(b,1,n,m)

        q = q * self.scale
        if attend_means:
            attn = (q @ key.transpose(-2, -1)) # b x h x n x k
            rel_pos = pos[:,None,:,:] - pos_orig[:,:,None,:] # b x n x k x d
        else:
            attn = (q*k).sum(-1) # b x h x n x m

            rel_pos = pos_orig.unsqueeze(2) - pos.reshape(b,n,m,d) # b x n x m x d
            '''
            rel_cluster_feat = cluster_feat_orig.unsqueeze(2) - cluster_feat.reshape(b,n,m,-1) # b x n x m x c_
            cluster_dist = (rel_cluster_feat**2).sum(-1) # b x n x m
            cluster_dist = cluster_dist.unsqueeze(1)
            attn = attn - cluster_dist
            '''
        
        pos_bias = self.pos_mlp(rel_pos).permute(0,3,1,2) # b x h x n x m / b x h x n x k
        attn = attn + pos_bias 
        if mask is not None:
            mask = (1-mask)*(-100) # 1->0, 0->-100
            attn = attn + mask
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if attend_means:
            feat = (attn @ v).permute(0,2,1,3).reshape(b,n,c)
        else:
            feat = (attn.unsqueeze(-1)*v).sum(3).permute(0,2,1,3).reshape(b,n,c)
        feat = self.proj(feat)
        feat = self.proj_drop(feat)

        if q_subsample_idx is None: 
            return feat
        else:
            return pos_orig, feat, mask_orig

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        return flops


class ClusterTransformerBlock(nn.Module):
    r""" Cluster Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        pos_dim: dimension of x,y coordinates
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        pos_mlp_bias: add learnable bias to pos mlp
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, pos_dim=2,
                 mlp_ratio=4., qkv_bias=True, pos_mlp_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = ClusterAttention(
            dim, num_heads=num_heads, pos_dim=pos_dim,
            qkv_bias=qkv_bias, pos_mlp_bias=pos_mlp_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, pos, feat, cluster_feat, cluster_score, mean_assignment, mask, member_idx, batch_idx, k, valid_row_idx, attend_means, cluster_mask=None):
        """
        Args:
            pos - b x n x d, the x,y position of points
            feat - b x n x c, the features of points
        """

        b,n,c = feat.shape
        d = pos.shape[2]
        assert c == self.dim, "dim does not accord to input"
        assert d == self.pos_dim, "pos dim does not accord to input"

        shortcut = feat 
        feat = self.norm1(feat)

        # cluster attention 
        feat = self.attn(pos, feat, cluster_feat, cluster_score, mean_assignment, mask, member_idx, batch_idx, k, valid_row_idx, attend_means, cluster_mask=cluster_mask)

        '''
        if (not attend_means) and (member_idx is not None):
            z,m=member_idx.shape
            if cluster_mask is not None:
                member_idx = member_idx * cluster_mask
                member_idx = (1-cluster_mask)*n + member_idx
            if valid_row_idx is not None:
                member_idx_ = member_idx.new(b*k,m).zero_() + n # elements from blank cluster will go to extra col
                member_idx_[valid_row_idx] = member_idx
                member_idx = member_idx_
                feat_ = feat.new(b*k,m,c).zero_()
                feat_[valid_row_idx] = feat
                feat = feat_
            member_idx = member_idx.reshape(b,-1) # b x k*m
            feat = feat.reshape(b,-1,c) # b x k*m x c
            from torch_scatter import scatter_mean
            new_feat = scatter_mean(index=member_idx.unsqueeze(-1).expand(-1,-1,c),dim=1,src=feat)
            feat = new_feat[:,:n].contiguous() # b x n x c
        '''

        # FFN
        feat = shortcut + self.drop_path(feat)
        feat = feat + self.drop_path(self.mlp(self.norm2(feat)))

        return feat, pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        return flops

class ClusterMerging(nn.Module):
    def __init__(self, dim, pos_dim, num_heads, drop=0., attn_drop=0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.weight_net = nn.Sequential(
                    nn.Linear(pos_dim,32),
                    nn.LayerNorm(32),
                    nn.GELU()
                )
        self.feat_net = nn.Sequential(
                    nn.Linear(dim,dim//3),
                    nn.GELU(),
                    nn.Linear(dim//3,1),
                    nn.Sigmoid()
                )
        self.linear = nn.Linear(dim*32,dim*2)

    def forward(self, pos, feat, mask, cluster_feat, mean_assignment, member_idx, batch_idx, k, valid_row_idx, cluster_mask):

        b,n,c = feat.shape
        d = pos.shape[2]
        nnc = mean_assignment.shape[-1]
        cluster_size=4
        z,m=member_idx.shape

        feat = self.norm(feat)

        # sample from q
        max_x = pos[:,:,0].max()+1
        max_y = pos[:,:,1].max()+1
        h = (torch.ceil(max_y / 2.0)*2).long().item() # make sure the number is even
        w = (torch.ceil(max_x / 2.0)*2).long().item()
        idx = torch.arange(n,device=pos.device).long().reshape(1,n,1).expand(b,-1,-1)
        idx = points2img(pos, idx, h, w) # b x 1 x h x w
        idx = idx[:,:,::2,::2].reshape(b,-1,1) # b x n' x 1
        n = idx.shape[1]
        '''
        idx = torch.randperm(n,device=feat.device)[:n//4]
        idx = idx.reshape(1,n//4,1).expand(b,-1,-1) # b x n' x 1
        '''

        mean_assignment = mean_assignment.gather(index=idx.expand(-1,-1,nnc),dim=1) # b x n' x nnc
        pos_orig = pos
        pos = pos.gather(index=idx.expand(-1,-1,d),dim=1) # b x n' x d
        feat_down = feat.gather(index=idx.expand(-1,-1,c),dim=1) # b x n' x c
        if mask is not None:
            mask= mask.gather(index=idx,dim=1) # b x n' x 1
        else:
            mask= None

        # randomly pick neighbors
        batch_idx2 = torch.arange(b,device=mean_assignment.device,dtype=mean_assignment.dtype).reshape(-1,1,1).expand(-1,n,nnc) # b x n x nnc
        member_idx = member_idx.reshape(b,k,m)[batch_idx2.reshape(-1),mean_assignment.reshape(-1)].reshape(b,n,nnc*m) # b x n x nnc*m
        self_idx_rotate = idx.reshape(b*n,1)
        self_idx = (member_idx.reshape(b*n,-1)==self_idx_rotate).nonzero(as_tuple=True)
        if cluster_mask is not None:
            cluster_mask = cluster_mask.reshape(b,k,m)[batch_idx2.reshape(-1),mean_assignment.reshape(-1)].reshape(b*n,nnc*m) # b*n x nnc*m
            cluster_mask[self_idx]=0
            # sample a subset of neighbors
            neighbor_idx = cluster_mask.to(feat.dtype).clamp(min=1e-5).multinomial(cluster_size-1).reshape(b,n,cluster_size-1) # b x n x cluster_size-1
            cluster_mask = cluster_mask.reshape(b,n,-1)
        else:
            ones = torch.ones_like(member_idx).reshape(b*n,-1)
            ones[self_idx]=1e-5
            neighbor_idx = ones.multinomial(cluster_size-1).reshape(b,n,cluster_size-1) # b x n x cluster_size-1
        member_idx = member_idx.gather(index=neighbor_idx,dim=-1) # b x n x m-1
        member_idx = torch.cat([member_idx,self_idx_rotate.reshape(b,n,1)],dim=-1) # b x n x cluster_size
        if cluster_mask is not None:
            cluster_mask = cluster_mask.gather(index=neighbor_idx,dim=-1) # b x n x m-1
            cluster_mask = torch.cat([cluster_mask,torch.ones(b,n,1,device=cluster_mask.device,dtype=cluster_mask.dtype)],dim=-1) # b x n x cluster_size
        batch_idx = torch.arange(b,device=mean_assignment.device,dtype=mean_assignment.dtype).reshape(-1,1,1).expand(-1,n,cluster_size) # b x n x m
        m = cluster_size

        member_idx = member_idx.reshape(-1) # z*m
        batch_idx = batch_idx.reshape(-1) # z*m
        feat = feat[batch_idx,member_idx].clone().reshape(b,n,m,c) # b x n x m x c
        pos_rel = pos_orig[batch_idx,member_idx].clone().reshape(b,n,m,d) # b x n x m x d
        pos_rel = pos_rel - pos.unsqueeze(2)
        weights = self.weight_net(pos_rel) # b x n x m x 32
        feat_rel = feat - feat_down.unsqueeze(2) # b x n x m x c
        feat_weights = self.feat_net(feat_rel) # b x n x m x 1
        weights = weights * feat_weights
        feat = (weights.permute(0,1,3,2) @ feat).view(b,n,-1) # b x n x 32c
        feat = self.linear(feat) # b x n x 2c



        '''
        # normalize
        pos_mean = pos.mean()
        pos_std = (pos.view(-1).var(dim=0, unbiased=False)+1e-5).pow(0.5)
        pos = (pos-pos_mean) / pos_std
        '''
        pos = pos.div(2,rounding_mode='floor')

        return pos, feat, mask



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, pos, feat, mask=None):
        """
        pos - b x n x 2
        feat - b x n x c
        mask - b x n x 1
        return
        pos - b x n x 2
        feat - b x n x c
        mask - b x n x 1
        """
        b,n,c = feat.shape
        assert c == self.dim, "dim does not accord to input"
        max_x = pos[:,:,0].max()+1
        max_y = pos[:,:,1].max()+1
        h = (torch.ceil(max_y / 2.0)*2).long().item() # make sure the number is even
        w = (torch.ceil(max_x / 2.0)*2).long().item()
        feat = points2img(pos, feat, h, w) # b x c x h x w
        if mask is not None:
            mask = points2img(pos, mask, h, w) # b x 1 x h x w
            feat *= mask
        x = feat

        x0 = x[:,:, 0::2, 0::2]
        x1 = x[:,:, 1::2, 0::2]
        x2 = x[:,:, 0::2, 1::2]
        x3 = x[:,:, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1).permute(0,2,3,1)  # b x h x w x 4*c

        x = self.norm(x)
        x = self.reduction(x)
        _,h,w,c = x.shape
        x = x.view(b,-1,c)

        # create new pos tensor
        pos = feat.new(b,h,w,2).zero_()
        hs = torch.arange(0,h)
        ws = torch.arange(0,w)
        ys,xs = torch.meshgrid(hs,ws)
        xs=xs.unsqueeze(0).expand(b,-1,-1)
        ys=ys.unsqueeze(0).expand(b,-1,-1)
        pos[:,:,:,0]=xs
        pos[:,:,:,1]=ys
        pos = pos.view(b,-1,2)

        # mask
        if mask is not None:
            mask = nn.AdaptiveMaxPool2d((h,w))(mask.float())
            mask = mask.view(b,-1).unsqueeze(2) # b x n x 1

        return pos, x, mask

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def flops(self):
        flops = 0
        return flops


class BasicLayer(nn.Module):
    """ A basic cluster Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        cluster_size: the avg cluster size
        max_cluster_size: maximum cluster size
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        pos_lambda: lambda for pos in k-means
        pos_dim: dimension of x,y coordinates
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        pos_mlp_bias: add learnable bias to pos mlp
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, cluster_size, max_cluster_size, depth, num_heads, pos_lambda=0.0003, pos_dim=2, 
                 mlp_ratio=4., qkv_bias=True, pos_mlp_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.pos_lambda = pos_lambda
        self.cluster_size=cluster_size
        self.max_cluster_size = None if max_cluster_size==0 else max_cluster_size
        self.pos_dim = pos_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.cluster_feat_mlp = nn.Linear(dim, head_dim, bias=True)

        # build blocks
        self.blocks = nn.ModuleList([
            ClusterTransformerBlock(dim=dim,
                                 num_heads=num_heads, pos_dim=pos_dim,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, pos_mlp_bias=pos_mlp_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            '''
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
            '''
            self.downsample = downsample(dim=dim, pos_dim=pos_dim, num_heads = num_heads, 
                    drop=0.0, attn_drop=0.0, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, pos, feat, mask=None):
        '''
        pos - b x n x d
        feat - b x n x c
        mask - b x n x 1
        '''
        b,n,d = pos.shape
        h = self.num_heads
        assert torch.isnan(feat).any()==False, "feat 1 nan "+str(n) 
        assert torch.isinf(feat).any()==False, "feat 1 inf "+str(n) 
        c = feat.shape[2]
        c_ = c // h
        assert self.cluster_size > 0, 'self.cluster_size must be positive'
        self.k = int(math.ceil(n / float(self.cluster_size)))
        k = self.k
        cluster_feat = self.cluster_feat_mlp(feat) # b x n x c_
        if self.k>1:
            # perform k-means
            with torch.no_grad():
                cluster_mean, mean_assignment, member_idx, cluster_mask= kmeans(cluster_feat, self.k, max_cluster_size=self.max_cluster_size,num_nearest_mean=2, num_iter=10, pos=pos, pos_lambda=self.pos_lambda, valid_mask=mask, init='random',balanced=True, fillup=False, normalize=True) # b x k x m
            _,k,m = member_idx.shape
            self.k=k
            batch_idx = torch.arange(b,device=feat.device).long().repeat_interleave(k*m) # b*k*m
            member_idx = member_idx.reshape(b*k,m)
            batch_idx = batch_idx.reshape(b*k,m)
            '''
            if valid_row_idx is not None and len(valid_row_idx.shape)>1:
                cluster_mask = valid_row_idx.reshape(-1,m)
                valid_row_idx=None
                z=b*k
            elif valid_row_idx is not None:
                z = len(valid_row_idx)
                member_idx = member_idx[valid_row_idx] # z x m
                batch_idx = batch_idx[valid_row_idx] # z x m
                cluster_mask=None
            else:
                z=b*k
                cluster_mask=None
            def normalize(x):
                mean = x.mean()
                std = (x.view(-1).var(dim=0, unbiased=False)+1e-5).pow(0.5)
                x = (x-mean) / std
                return x
            cluster_feat_ = cluster_feat[batch_idx,member_idx].clone().reshape(z,m,-1)
            cluster_pos = pos[batch_idx,member_idx].clone().reshape(z,m,-1)
            if cluster_mask is None:
                cluster_mean = cluster_feat_.mean(1,keepdim=True) # z x 1 x c_
                cluster_pos_mean = cluster_pos.mean(1,keepdim=True) # z x 1 x c_
            else:
                cluster_mean = cluster_feat_.sum(1,keepdim=True) # z x 1 x c_
                cluster_pos_mean = cluster_pos.sum(1,keepdim=True) # z x 1 x c_
                cluster_mean = cluster_mean / (cluster_mask.sum(1).view(z,1,1)+1e-5)
                cluster_pos_mean = cluster_pos_mean / (cluster_mask.sum(1).view(z,1,1)+1e-5)
            cluster_dist = ((cluster_feat_-cluster_mean)**2).sum(-1) # z x m
            cluster_pos_dist = ((cluster_pos-cluster_pos_mean)**2).sum(-1) # z x m
            cluster_dist = cluster_dist + (self.pos_lambda / d * c) * cluster_pos_dist
            cluster_dist = 1/(cluster_dist+1e-5)
            if cluster_mask is not None:
                cluster_dist = cluster_dist + (1-cluster_mask)*(-1000)
            cluster_score = F.softmax(cluster_dist,dim=-1) # z x m
            assert torch.isnan(cluster_score).any()==False, "cluster score nan"
            assert torch.isinf(cluster_score).any()==False, "cluster score inf"
            '''
            z=b*k
            cluster_score=None
            valid_row_idx = None
        else:
            member_idx = None
            batch_idx = None
            valid_row_idx = None
            z = b
            cluster_score = None

        for i_blk in range(len(self.blocks)):
            attend_means = i_blk % 2
            #attend_means = 0
            blk = self.blocks[i_blk]
            if self.use_checkpoint:
                feat, pos = checkpoint.checkpoint(pos, feat, cluster_feat, cluster_score, mean_assignment, mask, member_idx, batch_idx, k, valid_row_idx, attend_means = attend_means, cluster_mask=cluster_mask)
            else:
                feat, pos = blk(pos, feat, cluster_feat, cluster_score, mean_assignment, mask,  member_idx, batch_idx, k, valid_row_idx, attend_means = attend_means, cluster_mask=cluster_mask)
            assert torch.isnan(feat).any()==False, "feat nan after blk"
            assert torch.isinf(feat).any()==False, "feat inf after blk"

        if self.downsample is not None:
            #pos, feat, mask = self.downsample(pos, feat, mask)
            pos, feat, mask = self.downsample(pos, feat, mask, cluster_feat, mean_assignment, member_idx, batch_idx, k, valid_row_idx, cluster_mask)
        assert torch.isnan(feat).any()==False, "feat 4 nan"
        assert torch.isinf(feat).any()==False, "feat 4 inf"
        return pos, feat, mask

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

    def flops(self):
        flops = 0
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        b,c,h,w = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)  # b x n x c
        assert torch.isnan(self.proj.weight).any()==False, "weight nan"
        assert torch.isinf(self.proj.weight).any()==False, "weight inf"
        assert torch.isnan(self.proj.bias).any()==False, "bias nan"
        assert torch.isinf(self.proj.bias).any()==False, "bias inf"
        if self.norm is not None:
            x = self.norm(x)
        assert torch.isnan(x).any()==False, "feat 000 nan"
        assert torch.isinf(x).any()==False, "feat 000 inf"

        pos = x.new(b,h,w,2).zero_()
        hs = torch.arange(0,h)
        ws = torch.arange(0,w)
        ys,xs = torch.meshgrid(hs,ws)
        xs=xs.unsqueeze(0).expand(b,-1,-1)
        ys=ys.unsqueeze(0).expand(b,-1,-1)
        pos[:,:,:,0]=xs
        pos[:,:,:,1]=ys
        pos = pos.view(b,-1,2) #  b x n x 2

        '''
        # normalize
        pos_mean = pos.mean()
        pos_std = (pos.view(-1).var(dim=0, unbiased=False)+1e-5).pow(0.5)
        pos = (pos-pos_mean) / pos_std
        '''

        return pos, x, None

    def flops(self):
        flops=0
        return flops


class ClusterTransformer(nn.Module):
    """

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        pos_lambda (tuple(float)): lambda for pos in kmeans
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        pos_dim : dimension of position coordinates
        depths (tuple(int)): Depth of each Cluster Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        pos_mlp_bias (bool): If True, add a learnable bias to pos mlp. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, pos_dim=2, cluster_size=49, max_cluster_size=0, pos_lambda=[100.0,30.0,10.0,3.0], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, pos_mlp_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, 
                 #downsample=PatchMerging,
                 downsample=ClusterMerging,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               pos_dim=pos_dim,
                               cluster_size=cluster_size,
                               max_cluster_size=max_cluster_size,
                               pos_lambda=pos_lambda[i_layer],
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, pos_mlp_bias=pos_mlp_bias,qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=downsample if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        #self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    def forward_features(self, x):
        '''
        x - b x c x h x w
        '''
        pos, x, mask = self.patch_embed(x) # b x n x c, b x n x d
        x = self.pos_drop(x)
        gsms = list()

        for i_layer in range(len(self.layers)):
            layer = self.layers[i_layer]
            ret = layer(pos, x, mask)
            if len(ret) == 3:
                pos, x, mask = ret
            else:
                pos, x, mask, prob_loss= ret
                gsms.append(prob_loss)

        assert torch.isnan(x).any()==False, "feat after layers nan"
        assert torch.isinf(x).any()==False, "feat after layers inf"
        x = self.norm(x) # b x n x c
        #x = self.avgpool(x.transpose(1, 2))  # b x c x 1
        if mask is not None:
            from torch_scatter import scatter_mean
            x = scatter_mean(dim=1,src=x,index=mask.long())
            x = x[:,1].contiguous()
        else:
            x = x.mean(1)
        assert torch.isnan(x).any()==False, "feat after pool nan"
        assert torch.isinf(x).any()==False, "feat after pool inf"
        x = x.reshape(x.shape[0],-1)
        return x, gsms 

    def forward(self, x):
        x, gsms = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        return flops
