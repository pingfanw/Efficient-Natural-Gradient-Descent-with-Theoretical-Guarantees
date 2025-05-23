import torch
import torch.nn as nn

__all__ = [
    'vit_small',
    'vit_base',
    'vit_large',
    'vit_huge'
]

class PatchEmbedding(nn.Module):
    def __init__(self, input_size=32, patch_size=4, in_channels=3, embedding_dimension=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embedding_dimension, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)            # [B, E, H/ps, W/ps]
        x = x.flatten(2)            # [B, E, num_patches]
        x = x.transpose(1, 2)       # [B, num_patches, E]
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embedding_dimension=128, depth=6, num_heads=4, mlp_ratio=4., qkv_bias=False, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(input_size, patch_size, in_channels, embedding_dimension)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dimension))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dimension))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(embedding_dimension, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embedding_dimension)
        self.head = nn.Linear(embedding_dimension, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, C]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]
        return self.head(cls)

def vit_small(**kwargs):
    depth = 6
    embedding_dimension = 128
    num_heads = 4
    input_size = kwargs['input_size']
    in_channels = kwargs['in_channels']
    num_classes = kwargs['num_classes']
    return VisionTransformer(
        input_size=input_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embedding_dimension=embedding_dimension,
        depth=depth,
        num_heads=num_heads,
        qkv_bias=True
    )

def vit_base(**kwargs):
    depth = 12
    embedding_dimension = 768
    num_heads = 12
    mlp_ratio = kwargs['mlp_ratio']
    input_size = kwargs['input_size']
    patch_size = kwargs['patch_size']
    in_channels = kwargs['in_channels']
    num_classes = kwargs['num_classes']
    return VisionTransformer(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embedding_dimension=embedding_dimension,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True
    )

def vit_large(**kwargs):
    depth = 24
    embedding_dimension = 1024
    num_heads = 16
    mlp_ratio = kwargs['mlp_ratio']
    input_size = kwargs['input_size']
    patch_size = kwargs['patch_size']
    in_channels = kwargs['in_channels']
    num_classes = kwargs['num_classes']
    return VisionTransformer(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embedding_dimension=embedding_dimension,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True
    )

def vit_huge(**kwargs):
    depth = 32
    embedding_dimension = 1280
    num_heads = 16
    mlp_ratio = kwargs['mlp_ratio']
    input_size = kwargs['input_size']
    patch_size = kwargs['patch_size']
    in_channels = kwargs['in_channels']
    num_classes = kwargs['num_classes']
    return VisionTransformer(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embedding_dimension=embedding_dimension,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True
    )
