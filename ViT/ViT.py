
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):  ## Norm
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


#MLP
class FeedForward(nn.Module):  ## MLP
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]  ###
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # print("after MSA: ",out.shape) torch.Size([32, 67, 64])
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, mode):
        super().__init__()

        self.depth = depth

        self.layers = nn.ModuleList([])

        for _ in range(depth):  # 6次
            self.layers.append(nn.ModuleList([
                # 先对输入做LN,并传入attention
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(7**2 + 1, 7**2 + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):  #x:([64, 5, 60])

        if self.mode == 'ViT':
            for attn, ff in self.layers:  #
                x = attn(x, mask=mask)  #
                x = ff(x)



        elif self.mode == 'CAF':

            last_output = []

            nl = 0

            for attn, ff in self.layers:

                last_output.append(x)
                #print("x: ",x.shape)
                if nl > 1:
                    #print("nl: ",nl,"x: ",x.shape)
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)

                x = attn(x, mask=mask)

                x = ff(x)

                nl += 1

        return x
Num_class=21 # 9 16 21
class ViT(nn.Module):
    def __init__(self, num_classes=Num_class,patch_size=7, dim=285, depth=5, heads=4, dim_head=16, mlp_dim=8,
                 dropout=0.1, emb_dropout=0.1):#dim=103/200
        super(ViT, self).__init__()

        self.patch_dim = (patch_size) ** 2

        self.transformer = Transformer(dim, depth, heads=heads, dim_head=dim_head, mlp_head=mlp_dim, dropout=dropout, mode='ViT')

        self.pos_embedding = nn.Parameter(torch.empty(1, (self.patch_dim + 1), dim))  # num_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )


    def forward(self, x, mask=None):  # [64, 1, 30, 13, 13]

        x = rearrange(x.squeeze(1), 'b c h w -> b (h w) c')  # -->[64,49,60]

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [1,1,200]-->[64, 1, 200]
        #print('x:',x.shape,'cls:',cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # -->[64, 50, 60]  T

        x += self.pos_embedding

        x = self.dropout(x)

        #print("before_t: ",x.shape)

        x = self.transformer(x, mask)  # main game  [64, 5, 60] --> [64, 5, 64]  [64, 50, 60] [64,17,60]
        #print(x.shape)
        x = self.to_cls_token(x[:, 0, :])  # -->[64, 60]

        x = self.mlp_head(x)

        return x


