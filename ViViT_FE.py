import math
import logging
from functools import partial
from collections import OrderedDict
from this import s
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Block


class ViViT_FE(nn.Module):
    def __init__(self, in_chans=3, spatial_embed_dim=32, sdepth=4, tdepth=4, num_spat_tokens=20, num_temp_tokens=20,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, spat_op='cls',
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,  norm_layer=None, num_classes=26):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            in_chans (int): number of input channels, RGB videos have 3 chanels
            spatial_embed_dim (int): spatial patch embedding dimension 
            sdepth (int): depth of spatial transformer
            tdepth(int):depth of temporal transformer
            num_spat_tokens(int): number of tokens input to spatial transformer - s
            num_temp_tokens(int): numbe of frames or tokens with varying temoral indices - f
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            spat_op(string): Spatial Transformer output type - pool(Global avg pooling of encded features) or cls(Just CLS token)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        temporal_embed_dim = spatial_embed_dim   #### one temporal token embedding dimension is equal to one spatial patch embedding dim
        print("Spatial embed dimension",spatial_embed_dim)
        print("Temporal embed dim:", temporal_embed_dim)
        print("Drop Rate: ",drop_rate)
        print("Attn drop rate: ",attn_drop_rate)
        print("Drop path rate: ",drop_path_rate)

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, spatial_embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_spat_tokens+1, spatial_embed_dim)) #num joints + 1 for cls token
        self.spatial_cls_token= nn.Parameter(torch.zeros(1,1,spatial_embed_dim)) #spatial cls token patch embed
        
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_temp_tokens+1, temporal_embed_dim)) #additional pos embedding zero for class token
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, temporal_embed_dim)) #temporal class token patch embed - this token is used for final classification!
        self.pos_drop = nn.Dropout(p=drop_rate)

        sdpr = [x.item() for x in torch.linspace(0, drop_path_rate, sdepth)]  # stochastic depth decay rule
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=sdpr[i], norm_layer=norm_layer)
            for i in range(sdepth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=temporal_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])

        self.Spatial_norm = norm_layer(spatial_embed_dim)
        self.Temporal_norm = norm_layer(temporal_embed_dim)

        #Classification head
        self.class_head = nn.Sequential(
            nn.LayerNorm(spatial_embed_dim),
            nn.Linear(temporal_embed_dim, num_classes)
        )


    def Spatial_forward_features(self, x, spat_op='cls'):
        #spat_op: 'cls' output is CLS token, otherwise global average pool of attention encoded spatial features
        #Input shape: batch x frame x spatial tokens x tube size
        b,f,s,t = x.shape
        x = rearrange(x, 'b f s t  -> (b f) s t', ) #for spatial transformer, batch size if b*f
        x = self.Spatial_patch_to_embedding(x) #all input spatial tokens, outut: b x f x s x Se(spatial_embed)
        class_token=torch.tile(self.spatial_cls_token,(b*f,1,1)) #(B*F,1,1)
        x = torch.cat((x,class_token),dim=1) 
        #print("After concate x dim: ",x.shape) #(B*F,s+1,spatial_embed)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        c=x.shape[-1]
        ###Extract Class token head from the output
        cls_token = x[:,-1,:]
        cls_token = torch.reshape(cls_token,(b,f*c))
        x = x[:,:s,:]
        x = rearrange(x, '(b f) s Se -> b f (s Se)', f=f) #BxFx(sxSe)

        #Determine the output type from Spatial transformer
        if spat_op=='cls':
            return cls_token
        else:
            return x #!!!!!!!!!!!!!!!!!!!!!!!!ALERT: ADD GLOBAL AVG POOLING HERE!!!!!!!!!!!!!!!!!
        

    def Temporal_forward_features(self, x):
        
        b  = x.shape[0]
        class_token=torch.tile(self.temporal_cls_token,(b,1,1)) #(B,1,embed_dim)
        x = torch.cat((x,class_token),dim=1) #(B,F+1,embed_dim)
        x += self.Temporal_pos_embed

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)

        ###Extract Class token head from the output
        x = x[:,-1,:]
        x = x.view(b, -1) # (Batch_size, class_embedding_size)
        return x

    def forward(self, x):
        spat_op='cls'
        x,cls_token = self.Spatial_forward_features(x.)
        temp_cls_token = torch.unsqueeze(temp_cls_token,1) #Bx1xJC
        #print("Temp cls token: ",temp_cls_token.shape)
        x= torch.cat((x,temp_cls_token),dim=1)
        #print("Temopral transformer input: ",x.shape) #Bxf+1xJC
        
        x = self.forward_features(x)
        x = self.class_head(x)

        return F.log_softmax(x,dim=1) 

#model=ActRecogTransformer()
#inp=torch.randn((64,120,25,3))
#op=model(inp)
#print("Op shape: ",op.shape)