from functools import partial
from unittest.mock import patch
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Block,PBlock


'''
H =img height
W = img width
T = video time
tw = tubelet width
th = tubelet height
tt = tubelet time
h = H/th
w = W/tw # h*w: numner of tubelets with unique spatial index
nb = T/tt #number of blocks or tubelets with unique temporal index
'''

class Spatial_Perceiver(nn.Module):
    def __init__(self, spatial_embed_dim=64, sdepth=4, tdepth=4, vid_dim=(128,128,100), perceiver_query_dim=(128,64),
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, patch_dim = (3,4,4),
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=None, num_classes=26):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            in_chans (int): number of input channels, RGB videos have 3 chanels
            spatial_embed_dim (int): spatial patch embedding dimension
            sdepth (int): depth of spatial perceiver transformer
            tdepth(int):depth of temporal transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            patch(tuple): patch size (ch,h,w)
            vid_dim: Original video (H , W, T)
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        query_tokens = perceiver_query_dim[0]
        query_token_dim = perceiver_query_dim[1]
        temporal_embed_dim = query_token_dim   #### one temporal token embedding dimension is equal to one token
        print("Spatial embed dimension",spatial_embed_dim)
        print("Temporal embed dim:", temporal_embed_dim)
        print("Drop Rate: ",drop_rate)
        print("Attn drop rate: ",attn_drop_rate)
        print("Drop path rate: ",drop_path_rate)
        #print("Tubelet dim: ",tubelet_dim)

        #c,tt,th,tw = tubelet_dim
        #self.tubelet_dim = tubelet_dim
        c,h,w = patch_dim
        self.patch_dim = patch_dim
        patch_size  =c*h*w
        ###Spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(patch_size,spatial_embed_dim) #nn.Conv3d(c, spatial_embed_dim, self.tubelet_dim[1:],
                                        #stride=self.tubelet_dim[1:],padding='valid',dilation=1)
        num_spat_tokens =  (vid_dim[0]*vid_dim[1]*c) // (patch_size) #(vid_dim[0]//th) * (vid_dim[1]//tw)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_spat_tokens, spatial_embed_dim)) #num joints + 1 for cls token

        self.tubelet_query = nn.Parameter(torch.zeros(1,query_tokens,query_token_dim)) #Learnable query vector to attend to all spatial tubelets. N x D
        self.query_pos_embed = nn.Parameter(torch.zeros(1,query_tokens,query_token_dim)) #num of query_tokens, query_tokens_dim
        
        num_temp_tokens = vid_dim[-1] #vid_dim[-1] // tt
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_temp_tokens+1, temporal_embed_dim)) #additional pos embedding zero for class token
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, temporal_embed_dim)) #temporal class token patch embed - this token is used for final classification!
        self.pos_drop = nn.Dropout(p=drop_rate)

        sdpr = [x.item() for x in torch.linspace(0, drop_path_rate, 1)]  # stochastic depth decay rule
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            PBlock(
                dim=spatial_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=sdpr[i], norm_layer=norm_layer)
            for i in range(1)])
        self.sdepth = sdepth

        self.blocks = nn.ModuleList([
            Block(
                dim=temporal_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])

        self.Spatial_norm = norm_layer(spatial_embed_dim)
        self.Temporal_norm = norm_layer(temporal_embed_dim)

        #Classification head
        self.class_head = nn.Sequential(
            nn.LayerNorm(temporal_embed_dim),
            nn.Linear(temporal_embed_dim, num_classes)
        )

    def Spatial_Perceiver_forward_features(self, x):
        #Input shape: batch x time x num_patches, patch_size
        b,t,np,ps = x.shape
        x = rearrange(x, 'b t np ps -> (b t) np ps', ) #for spatial transformer, batch size if b*f
        x = self.Spatial_patch_to_embedding(x) #all input spatial tokens, op: (b nc) x H/h x W/w x Se

        #print("Spatial embedded op: ",x.shape)
        #Add pos embedding and drop                        
        x += self.Spatial_pos_embed 
        x = self.pos_drop(x)

        #Query tokens
        latent_query = torch.tile(self.tubelet_query,((b*t),1,1)) #Replicate query batch times, B x query_tokens, query_token_dim

        #Add pos embeddings to latent query
        latent_query += self.query_pos_embed

        #Pass through transformer blocks
        blk = self.Spatial_blocks[0]
        for _ in range(self.sdepth):
            latent_query = blk(xq=latent_query, xkv=x)

        x = latent_query
        x = self.Spatial_norm(x)

        x = torch.mean(x, dim=1) #B x query_embed_dim
        x = torch.reshape(x, (b,t,-1))
        return x #b x t x Se

    def Temporal_forward_features(self, x):
        
        b  = x.shape[0]
        class_token=torch.tile(self.temporal_cls_token,(b,1,1)) #(B,1,temp_embed_dim)
        x = torch.cat((x,class_token),dim=1) #(B,F+1,temp_embed_dim)
        
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
        #Input x: batch x num_chans x video_time x img_height x img_width 

        b, ch, t, H, W = x.shape
        
        #Reshape frames into patches
        patch_size = self.patch_dim[0]*self.patch_dim[1]*self.patch_dim[2]
        x = torch.reshape(x,(b,t,-1,patch_size))
        #print("Spatial Perceiver input shape: ",x.shape)

        #Reshape input to pass through Conv3D patch embedding
        x = self.Spatial_Perceiver_forward_features(x) # input:  b x nc x ch x H x W x t, op: b x nc x query_token_dim
        x = self.Temporal_forward_features(x) #input: b x nc x query_token_dim, op: b x temporal_embed_dim
        x = self.class_head(x)
        return x #F.log_softmax(x,dim=1) 

'''
model=Spatial_Perceiver()
inp=torch.randn((2, 1, 3, 100, 128 , 128))
op=model(inp)
print("Op shape: ",op.shape)
'''
