import torch
import torch.nn as nn
# from typing import List, Dict, Tuple


class CoordinateImageBuilder:
    def __init__(self, **model_params):
        self.w = self.h = model_params['patch_size']
        self.problem_size = None
        self.total_nodes = None
        self.W = None
        self.H = None

    def normalize_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        batch_size = coords.shape[0]
        self.total_nodes = coords.shape[1]
        
        self.W = self.H = int(torch.ceil(torch.tensor(
            10 * (self.total_nodes ** 0.5) / self.w)).item()) * self.w

        scale = torch.tensor([self.W-1, self.H-1], device=coords.device)

        norm_coords = torch.floor(coords * scale)
        return norm_coords.long()

    def build_gtsp_image(self, node_xy: torch.Tensor, cluster_idx: torch.Tensor) -> torch.Tensor:

        batch_size = node_xy.shape[0]
        
        norm_coords = self.normalize_coordinates(node_xy)
        
        image = torch.full((batch_size, 1, self.H, self.W), 0.0,
                           device=node_xy.device)
        
        x_coords = norm_coords[..., 0].long().clamp(0, self.W - 1)  # 宽度
        y_coords = norm_coords[..., 1].long().clamp(0, self.H - 1)  # 高度
        
        batch_indices = torch.arange(batch_size, device=norm_coords.device)[:, None].expand(-1, self.total_nodes)
        
        channel_indices = torch.zeros_like(batch_indices)

        values = cluster_idx.float() + 1

        image[batch_indices.flatten(), 
              channel_indices.flatten(),
              y_coords.flatten(), 
              x_coords.flatten()] = values.flatten()
        
        return image


class PatchEmbedding(nn.Module):
    def __init__(self, **model_param):
        super().__init__()
        self.patch_size = model_param['patch_size']
        self.in_channels = model_param['in_channels']
        self.embedding_dim = model_param['embedding_dim']
        

        self.proj = nn.Conv2d(in_channels=self.in_channels, 
                             out_channels=self.embedding_dim,
                             kernel_size=self.patch_size, 
                             stride=self.patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, H, W)
        Returns:
            (batch_size, num_patches, embed_dim)
        """
        x = self.proj(x)        # [batch_size, embed_dim, H//patch_size, W//patch_size]

        x = x.flatten(2)        # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (batch_size, num_patches, embed_dim)
        return x


class ScalablePositionalEncoding(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.mlp = nn.Sequential(
            nn.Linear(2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_patches, embed_dim)
        Returns:
             (batch_size, num_patches, embed_dim)
        """
        batch_size, num_patches, _ = x.shape

        H = int(num_patches ** 0.5)
        W = num_patches // H
        if H * W < num_patches:
            W += 1

        grid_h = torch.linspace(0, 1, H, device=x.device)
        grid_w = torch.linspace(0, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        
        # [H*W, 2]
        grid = torch.stack([grid_w, grid_h], dim=-1).reshape(-1, 2)

        grid = grid[:num_patches]

        pos_embed = self.mlp(grid)  # [num_patches, embedding_dim]
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pos_embed


class TransformerEncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.num_heads = model_params['head_num']
        self.self_attn = nn.MultiheadAttention(self.embedding_dim, self.num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        )

        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_patches, embedding_dim]
        Returns:
            [batch_size, num_patches, embedding_dim]
        """
        # self-attention
        residual = x
        x = x.transpose(0, 1)  # [num_patches, batch_size, embedding_dim]
        x, _ = self.self_attn(x, x, x)
        x = x.transpose(0, 1)  # [batch_size, num_patches, embedding_dim]
        x = residual + x
        x = self.norm1(x)
        
        # Feed-forward
        residual = x
        x = self.ffn(x)
        x = residual + x
        x = self.norm2(x)
        
        return x


class VisionTransformer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.patch_size = model_params['patch_size']
        self.in_channels = model_params['in_channels']
        self.embedding_dim = model_params['embedding_dim']
        
        self.patch_embed = PatchEmbedding(**model_params)
        
        self.pos_embed = ScalablePositionalEncoding(**model_params)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(**model_params)
            for _ in range(int(model_params['encoder_layer_num']))
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, H, W)
        Returns:
            (batch_size, num_patches, embed_dim)
        """
        # patch embedding
        x = self.patch_embed(x)
        
        # apply positional encoding
        pos_embed = self.pos_embed(x)
        x = x + pos_embed
        
        # apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        return x