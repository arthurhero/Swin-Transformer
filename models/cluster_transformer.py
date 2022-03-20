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
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .pointconv_utils import points2img, kmeans_keops, cluster2points, points2cluster
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
        self.pos_mlp = nn.Linear(pos_dim, num_heads, bias=pos_mlp_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pos, feat, mask):
        """
        Args:
            pos - k x m x 2, the x,y position of points, k is the total number of clusters in all batches, m is the largest size of any cluster
            feat - k x m x c, the features of points
            mask - k x m x 1, the binary mask indicating valid points
        """
        k_,m,c = feat.shape
        d = pos.shape[2]
        assert c == self.dim, "dim does not accord to input"
        assert d == self.pos_dim, "pos dim does not accord to input"
        qkv = self.qkv(feat).reshape(k_ ,m, 3, self.num_heads, c//self.num_heads).permute(2, 0, 3, 1, 4) # 3 x k x h x m x c'
        q, k, v = qkv[0], qkv[1], qkv[2]  # k x h x m x c'

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # k x h x m x m

        # calculate bias for pos
        pos = pos.clone().to(feat.dtype)
        for i in range(d):
            pos[:,:,i] = pos[:,:,i].clone() / pos[:,:,i].max() # normalize
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2) # k x m x m x 2
        pos_bias = self.pos_mlp(rel_pos) # k x m x m x h
        pos_bias = pos_bias.permute(0,3,1,2) # k x h x m x m

        attn = attn + pos_bias 
        if mask is not None:
            mask = mask.permute(0,2,1).unsqueeze(2) # k x 1 x 1 x m
            mask = (1-mask)*(-100) # 1->0, 0->-100

            attn = attn + mask
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(k_, m, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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

    def forward(self, pos, feat, mask):
        """
        Args:
            pos - k x m x 2, the x,y position of points, k is the total number of clusters in all batches, m is the largest size of any cluster
            feat - k x m x c, the features of points
            mask - k x m x 1, the binary mask indicating valid points
        """

        k,m,c = feat.shape
        d = pos.shape[2]
        assert c == self.dim, "dim does not accord to input"
        assert d == self.pos_dim, "pos dim does not accord to input"

        shortcut = feat 
        x = self.norm1(feat)

        # cluster attention 
        x = self.attn(pos, x, mask)  # k x m x c

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if mask is not None:
            x *= mask # make sure invalid points are 0

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
        pos_dim: dimension of x,y coordinates
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, pos_dim=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        assert pos_dim == 2, "regular downsample only works for regular images"
        self.pos_dim = pos_dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, pos, feat, mask=None):
        """
        pos - b x 2 x n
        feat - b x c x n
        mask - b x 1 x n
        return
        pos - b x 2 x n
        feat - b x c x n
        mask - b x 1 x n
        """
        #assert mask is None, "irregular image should not call patch merge"
        b,c,n = feat.shape
        max_x = pos[:,0].max()+1
        max_y = pos[:,1].max()+1
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
        x = self.reduction(x).permute(0,3,1,2)
        _,c,h,w = x.shape
        x = x.view(b,c,-1)

        # create new pos tensor
        pos = feat.new(b,2,h,w).zero_().long()
        hs = torch.arange(0,h).long()
        ws = torch.arange(0,w).long()
        ys,xs = torch.meshgrid(hs,ws)
        xs=xs.unsqueeze(0).expand(b,-1,-1)
        ys=ys.unsqueeze(0).expand(b,-1,-1)
        pos[:,0,:,:]=xs
        pos[:,1,:,:]=ys
        pos = pos.view(b,2,-1)

        # mask
        if mask is not None:
            mask = nn.AdaptiveMaxPool2d((h,w))(mask.float()).long()
            mask = mask.view(b,1,-1)

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
        k: number of clusters in k-means
        cluster_size: the avg cluster size (priority above k)
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

    def __init__(self, dim, k, cluster_size, max_cluster_size, depth, num_heads, pos_lambda=0.0003, pos_dim=2, 
                 mlp_ratio=4., qkv_bias=True, pos_mlp_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.pos_lambda = pos_lambda
        self.k=k
        self.cluster_size=cluster_size
        self.max_cluster_size = None if max_cluster_size==0 else max_cluster_size
        self.pos_dim = pos_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

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
            self.downsample = downsample(dim=dim, pos_dim=pos_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, pos, feat, mask=None):
        '''
        pos - b x 2 x n
        feat - b x c x n
        mask - b x 1 x n
        '''
        kmeans_after_blk = True 
        b,d,n = pos.shape
        c = feat.shape[1]
        if self.k == 0 or self.cluster_size != 0:
            self.k = int(math.ceil(n / float(self.cluster_size)))
        if self.k>1:
            # perform k-means
            with torch.no_grad():
                _, _, member_idx, cluster_mask = kmeans_keops(feat, self.k, max_cluster_size=self.max_cluster_size,num_nearest_mean=1, num_iter=10, pos=pos, pos_lambda=self.pos_lambda, valid_mask=mask, init='random') # b x c x k, b x 1 x n, b x m x k, b x m x k
                #print("keep ratio", cluster_mask.sum() / (b*n))
            cluster_pos, cluster_feat, cluster_mask, valid_row_idx = points2cluster(pos, feat, member_idx, cluster_mask)
        else:
            cluster_pos = pos.permute(0,2,1)
            cluster_feat = feat.permute(0,2,1)
            if mask is None:
                cluster_mask = None 
            else:
                cluster_mask = mask.permute(0,2,1)

        k = self.k
        for i_blk in range(len(self.blocks)):
            blk = self.blocks[i_blk]
            if self.use_checkpoint:
                cluster_feat = checkpoint.checkpoint(cluster_pos, cluster_feat, cluster_mask)
            else:
                cluster_feat = blk(cluster_pos, cluster_feat, cluster_mask)

        if self.k==1:
            new_pos = cluster_pos.permute(0,2,1)
            new_feat = cluster_feat.permute(0,2,1)
            if self.downsample is not None:
                new_pos, new_feat, mask = self.downsample(new_pos, new_feat, mask)
            return new_pos, new_feat, mask

        # convert back to batches
        new_pos, new_feat, new_mask = cluster2points(cluster_pos, cluster_feat, cluster_mask, valid_row_idx, b, self.k, filter_invalid=True)
        '''
        if new_mask is not None:
            print('min kept point ratio before ds',new_mask.sum(-1).min() / n)
            print('avg kept point ratio before ds',new_mask.sum() / (b*n))
            '''

        if self.downsample is not None:
            new_pos, new_feat, new_mask = self.downsample(new_pos, new_feat, new_mask)
        return new_pos, new_feat, new_mask

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

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        b,c,h,w = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)  # b x n x c
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2) # b x c x n

        pos = x.new(b,2,h,w).zero_().long()
        hs = torch.arange(0,h).long()
        ws = torch.arange(0,w).long()
        ys,xs = torch.meshgrid(hs,ws)
        xs=xs.unsqueeze(0).expand(b,-1,-1)
        ys=ys.unsqueeze(0).expand(b,-1,-1)
        pos[:,0]=xs
        pos[:,1]=ys
        pos = pos.view(b,2,-1) #  b x 2 x n

        return pos, x, None

    def flops(self):
        flops=0
        return flops


class ClusterTransformer(nn.Module):
    """

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        k (tuple(int)): Number of kmeans clusters
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
                 embed_dim=96, pos_dim=2, k=[64, 16, 4, 1], cluster_size=49, max_cluster_size=0, pos_lambda=[0.0003, 0.0001, 0.00003], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, pos_mlp_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, downsample=PatchMerging,
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
            layer = BasicLayer(dim=int(embed_dim * 2 ** min(i_layer,2)),
                               #dim=int(embed_dim * 2 ** i_layer),
                               pos_dim=pos_dim,
                               k=k[i_layer],
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
        pos, x, mask = self.patch_embed(x) # b x c x n, b x d x n
        x = self.pos_drop(x)

        for i_layer in range(len(self.layers)):
            layer = self.layers[i_layer]
            pos, x, mask = layer(pos, x, mask)

        x = self.norm(x.permute(0,2,1)) # b x n x c
        x = self.avgpool(x.transpose(1, 2))  # b x c x 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        return flops
