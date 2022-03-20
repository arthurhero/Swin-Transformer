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
from .point_utils import points2img, kmeans_keops, cluster2points, points2cluster, batched_bincount
import torch_scatter

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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

    def __init__(self, dim, num_heads, pos_dim=2, qkv_bias=True, pos_mlp_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #self.qkv = torch.nn.Conv1d(dim, dim * 3, 1, stride=1, groups=num_heads, bias=qkv_bias)
        #self.pos_mlp = nn.Linear(pos_dim, num_heads, bias=pos_mlp_bias)
        self.pos_mlp = torch.nn.Conv1d(num_heads*pos_dim, num_heads, 1, stride=1, groups=num_heads, bias=pos_mlp_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pos, feat, member_idx, cluster_mask):
        """
        Args:
            pos - b x n x d, the x,y position of points, k is the total number of clusters in all batches, m is the largest size of any cluster
            feat - b x n x c, the features of points
            member_idx - b x h x k x m
            cluster_mask - b*h x k x m
        """
        b,n,c = feat.shape
        if member_idx is not None:
            _,h,k,m = member_idx.shape
            '''
            _,k,m = member_idx.shape
            h = self.num_heads
            '''
        else:
            h = self.num_heads
            k = 1
            m = n
        d = pos.shape[2]
        assert c == self.dim, "dim does not accord to input"
        assert d == self.pos_dim, "pos dim does not accord to input"
        c_ = c // h
        #qkv = self.qkv(feat.reshape(b*n,c,1)).reshape(b,n,h,3,c_) # b x n x h x 3 x c_
        qkv = self.qkv(feat).reshape(b,n,h,3,c_) # b x n x h x 3 x c_
        '''
        '''
        if member_idx is not None:
            qkv = qkv.permute(3,0,2,1,4).reshape(3,b*h,n,c_) # 3 x b*h x n x c_
            member_idx = member_idx.reshape(-1) # b*h*k*m
            batch_idx = torch.arange(b*h,device=feat.device).long().unsqueeze(1).expand(-1,k*m).reshape(-1) # b*h*k*m
            qkv = qkv[:,batch_idx,member_idx].clone() # 3 x b*h*k*m x c_
            qkv = qkv.reshape(3,b,h,k,m,c_).permute(0,1,3,2,4,5).reshape(3,b*k,h,m,c_)
            '''
            member_idx = member_idx.reshape(-1) # b*k*m
            batch_idx = torch.arange(b,device=feat.device).long().unsqueeze(1).expand(-1,k*m).reshape(-1) # b*k*m
            feat = feat[batch_idx,member_idx].clone().reshape(b*k,m,c)
            '''
        else:
            qkv = qkv.permute(3,0,2,1,4) # 3 x b x h x n x c_
            '''
            feat = feat
            '''
        #qkv = self.qkv(feat).reshape(b*k,m,h,3,c_).permute(3,0,2,1,4) # 3 x b*k x h x m x c_

        q, key, v = qkv[0], qkv[1], qkv[2]  # b*k x h x m x c_

        q = q * self.scale
        attn = (q @ key.transpose(-2, -1)) # b*k x h x m x m

        # calculate bias for pos
        pos = pos.to(feat.dtype)
        pos = pos / pos.view(-1,d).max(0)[0] # normalize
        pos = pos.unsqueeze(1).expand(-1,h,-1,-1)
        if member_idx is not None:
            pos = pos.reshape(b*h,n,d)[batch_idx,member_idx].clone() # b*h*k*m x d
            pos = pos.reshape(b,h,k,m,d).permute(0,2,1,3,4).reshape(b*k,h,m,d)
        '''
        if member_idx is not None:
            pos = pos[batch_idx,member_idx].clone().reshape(b,k,m,d).reshape(b*k,m,d)
        '''
        rel_pos = pos.unsqueeze(2) - pos.unsqueeze(3) # b*k x h x m x m x d
        #rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2) # b*k x m x m x d
        rel_pos = rel_pos.permute(0,2,3,1,4).reshape(b*k*m*m,h*d,1)
        pos_bias = self.pos_mlp(rel_pos).reshape(b*k,m,m,h).permute(0,3,1,2) # b*k x h x m x m
        '''
        pos_bias = self.pos_mlp(rel_pos).permute(0,3,1,2) # b*k x h x m x m
        '''

        attn = attn + pos_bias 
        if cluster_mask is not None:
            mask = cluster_mask.reshape(b,h,k,m).permute(0,2,1,3).reshape(b*k,h,1,m)
            #mask = cluster_mask.reshape(b*k,1,1,m)
            '''
            mask = (mask.expand(-1,-1,m,-1) + torch.eye(m,device=mask.device,dtype=mask.dtype))
            mask = mask - (mask==2).to(mask.dtype)
            assert mask.max()<=1 and mask.min()>=0, "not only 0 and 1 in mask"
            assert mask.sum(-1).sum(-1).min()>0, "there should be 1 in every cluster!"
            '''
            mask = (1-mask)*(-100) # 1->0, 0->-100
            attn = attn + mask
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        feat = (attn @ v).reshape(b,k,h,m,c_).permute(0,2,1,3,4).reshape(b*h*k*m,c_) # b*h*k*m x c_
        #feat = (attn @ v).reshape(b,k,h,m,c_).permute(0,1,3,2,4).reshape(b*k*m,c) # b*k*m x c
        if member_idx is not None:
            if cluster_mask is not None:
                valid_idx = cluster_mask.reshape(-1).nonzero().reshape(-1)
                batch_idx = batch_idx[valid_idx]
                member_idx = member_idx[valid_idx]
                feat = feat[valid_idx]
            new_feat = torch.zeros(b*h,n,c_, device=feat.device, dtype=feat.dtype)
            new_feat[batch_idx, member_idx] = feat
            feat = new_feat.reshape(b,h,n,c_).permute(0,2,1,3).reshape(b,n,c) # b x n x c
            '''
            new_feat = torch.zeros(b,n,c, device=feat.device, dtype=feat.dtype)
            new_feat[batch_idx, member_idx] = feat
            feat = new_feat
            '''
        else:
            feat = feat.reshape(b,h,n,c_).permute(0,2,1,3).reshape(b,n,c)
            #feat = feat.reshape(b,n,c)
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        return feat

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

    def forward(self, pos, feat, member_idx, cluster_mask):
        """
        Args:
            pos - b x n x d, the x,y position of points, k is the total number of clusters in all batches, m is the largest size of any cluster
            feat - b x n x c, the features of points
            member_idx - b x h x k x m
            cluster_mask - b*h x k x m
        """

        b,n,c = feat.shape
        d = pos.shape[2]
        assert c == self.dim, "dim does not accord to input"
        assert d == self.pos_dim, "pos dim does not accord to input"

        shortcut = feat 
        x = self.norm1(feat)

        # cluster attention 
        x = self.attn(pos, x, member_idx, cluster_mask)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        return flops


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


class ClusterMerging(nn.Module):
    r""" Cluster Merging Layer.

    Args:
        dim (int): Number of input channels.
        pos_dim: dimension of x,y coordinates
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, pos_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim 
        self.norm = norm_layer(dim+pos_dim)
        self.linear = nn.Linear(dim+pos_dim, 2*dim, bias=False)
        '''
        self.act = nn.GELU()
        self.linear2 = nn.Linear(2*pos_dim, dim*2*dim, bias=True)
        '''

    def forward(self, pos, feat, mask=None):
        """ 
        pos - b x n x d
        feat - b x n x c
        mask - b x n x 1
        return
        pos - b x n x d
        feat - b x n x c
        mask - b x n x 1
        """
        b,n,c = feat.shape
        assert c == self.dim, "dim does not accord to input"
        d = pos.shape[2]
        if mask is not None:
            pos = pos *  mask
            feat = feat * mask
        # kmeans on feat
        k = int(math.ceil(n / 4.0)) # avg cluster size is 4
        pos_lambda = 100.0
        with torch.no_grad():
            _, mean_assignment, member_idx, cluster_mask = kmeans_keops(feat, k, num_nearest_mean=1, num_iter=10, pos=pos, pos_lambda=pos_lambda, valid_mask=mask, init='random') # b x n x 1, b x k x m, b x k x m
        m = member_idx.shape[2]
        pos = pos.to(feat.dtype)
        pos = pos / pos.view(-1,d).max(0)[0] # normalize
        mean_pos = torch.zeros(b,k,d,device = pos.device,dtype=pos.dtype)
        mean_pos.scatter_add_(dim=1, index=mean_assignment.expand(-1,-1,d), src=pos)
        if mask is not None:
            count = batched_bincount(mean_assignment.squeeze(2), mask.squeeze(2), k) # b x k
        else:
            count = batched_bincount(mean_assignment.squeeze(2), None, k) # b x k
        mean_pos = mean_pos / count.unsqueeze(2) # b x k x d
        rel_pos = pos - mean_pos.gather(index=mean_assignment.expand(-1,-1,d), dim=1) # b x n x d

        feat = self.norm(torch.cat([feat, rel_pos], dim=-1)) # b x n x (2+c)

        feat = self.linear(feat) # b x n x 2c
        
        cluster_feat, cluster_mask, valid_row_idx = points2cluster(feat, None, member_idx, cluster_mask) # k' x m x c
        count2 = cluster_mask.sum(1) # k' x 1
        if valid_row_idx is not None:
            count = count.view(-1)[valid_row_idx]
        assert (count2.view(-1) == count.view(-1)).all(), 'counts does not fit'
        count = count2

        merged_feat = cluster_feat / count.unsqueeze(2)
        merged_feat = merged_feat.sum(1) # k' x 2c

        # convert back to batch shape 
        if valid_row_idx is None:
            new_mask = None
            new_feat = merged_feat.reshape(b,k,2*c)
            new_pos = mean_pos
        else:
            new_mask = torch.zeros(b*k,1,device = feat.device, dtype=merged_feat.dtype)
            new_mask[valid_row_idx] = 1 
            new_feat = torch.zeros(b*k,2*c,device = feat.device, dtype=merged_feat.dtype)
            new_feat[valid_row_idx] = merged_feat
            new_pos = torch.zeros(b*k,d,device = feat.device, dtype=mean_pos.dtype)
            new_pos[valid_row_idx] = mean_pos.reshape(-1,d)[valid_row_idx]

            new_mask = new_mask.reshape(b,k,1)
            new_feat = new_feat.reshape(b,k,2*c)
            new_pos = new_pos.reshape(b,k,d)

        assert torch.isnan(new_pos).any() == False, "new pos has nan"
        assert torch.isinf(new_pos).any() == False, "new pos has inf"
        assert torch.isnan(new_feat).any() == False, "new feat has nan"
        assert torch.isinf(new_feat).any() == False, "new feat has inf"
        if new_mask is not None:
            assert torch.isnan(new_mask).any() == False, "new mask has nan"
            assert torch.isinf(new_mask).any() == False, "new mask has inf"
        return new_pos, new_feat, new_mask

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
            if downsample == PatchMerging:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer)
            else:
                self.downsample = downsample(dim=dim, pos_dim=pos_dim, norm_layer=norm_layer)
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
        feat_h = feat.reshape(b,n,h,c_).permute(0,2,1,3).reshape(-1,n,c_) # (b*h) x n x c_
        pos_h = pos.unsqueeze(1).expand(-1,h,-1,-1).reshape(-1,n,d) # (b*h) x n x d
        if mask is not None:
            mask_h = mask.unsqueeze(1).expand(-1,h,-1,-1).reshape(-1,n,1) # (b*h) x n x 1
        else:
            mask_h = None
        '''
        '''
        if self.k>1:
            # perform k-means
            with torch.no_grad():
                _, _, member_idx, cluster_mask = kmeans_keops(feat_h, self.k, max_cluster_size=self.max_cluster_size,num_nearest_mean=1, num_iter=10, pos=pos_h, pos_lambda=self.pos_lambda, valid_mask=mask_h, init='random') # (b*h) x k x m, (b*h) x k x m
                #_, _, member_idx, cluster_mask = kmeans_keops(feat, self.k, max_cluster_size=self.max_cluster_size,num_nearest_mean=1, num_iter=10, pos=pos, pos_lambda=self.pos_lambda, valid_mask=mask, init='random') # b x k x m, b x k x m
            #print("zero cluster num", b*h*k - len(cluster_mask.sum(-1).reshape(-1).nonzero().reshape(-1)))
            member_idx = member_idx.reshape(b,h,k,member_idx.shape[-1])
        else:
            member_idx = None
            #cluster_mask = mask
            '''
            '''    
            cluster_mask = mask_h
            if mask_h is not None:
                cluster_mask = mask_h.transpose(1,2) # (b*h) x 1 x n

        for i_blk in range(len(self.blocks)):
            blk = self.blocks[i_blk]
            if self.use_checkpoint:
                feat = checkpoint.checkpoint(pos, feat, member_idx, cluster_mask)
            else:
                feat = blk(pos, feat, member_idx, cluster_mask)

        if self.downsample is not None:
            pos, feat, mask = self.downsample(pos, feat, mask)
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
                 downsample=PatchMerging,
                 #downsample=ClusterMerging,
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
        self.avgpool = nn.AdaptiveAvgPool1d(1)
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
        x = self.avgpool(x.transpose(1, 2))  # b x c x 1
        assert torch.isnan(x).any()==False, "feat after pool nan"
        assert torch.isinf(x).any()==False, "feat after pool inf"
        x = torch.flatten(x, 1)
        return x, gsms 

    def forward(self, x):
        x, gsms = self.forward_features(x)
        x = self.head(x)
        return x, gsms 

    def flops(self):
        flops = 0
        return flops
