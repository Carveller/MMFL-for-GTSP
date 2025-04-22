import torch
import torch.nn as nn
# from typing import List, Dict, Tuple


class CoordinateImageBuilder:
    """构建GTSP问题的坐标图像"""
    def __init__(self, **model_params):
        """
        初始化构建器
        """
        # 根据问题规模自适应调整分辨率
        self.w = self.h = model_params['patch_size']    # 固定patch大小
        self.problem_size = None
        self.total_nodes = None
        self.W = None
        self.H = None

    def normalize_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        将[0,1]范围的坐标归一化到图像尺寸
        Args:
            coords: shape为(batch, problem_size+1, 2)的张量，最后一维为(x,y)坐标
        Returns:
            归一化后的整数坐标，shape为(batch, problem_size+1, 2)
        """
        batch_size = coords.shape[0]
        self.total_nodes = coords.shape[1]
        
        # 计算自适应分辨率
        self.W = self.H = int(torch.ceil(torch.tensor(
            10 * (self.total_nodes ** 0.5) / self.w)).item()) * self.w
        
        # 为避免边界问题，使用缩放因子0.95
        scale = torch.tensor([self.W-1, self.H-1], device=coords.device)
        # 保持批次维度，直接归一化
        norm_coords = torch.floor(coords * scale)
        return norm_coords.long()

    def build_gtsp_image(self, node_xy: torch.Tensor, cluster_idx: torch.Tensor) -> torch.Tensor:
        """
        构建GTSP的单通道坐标图像
        Args:
            node_xy: shape为(batch_size, problem_size+1, 2)的张量
            cluster_idx: shape为(batch_size, problem_size+1)的张量
        Returns:
            shape为(batch_size, 1, H, W)的坐标图像张量
        """
        batch_size = node_xy.shape[0]
        
        # 归一化坐标 [batch_size, total_nodes, 2]
        norm_coords = self.normalize_coordinates(node_xy)
        
        # 创建空白图像，填充0作为背景
        image = torch.full((batch_size, 1, self.H, self.W), 0.0,
                           device=node_xy.device)
        
        # 获取已经归一化的坐标 - 注意xy对应宽高
        x_coords = norm_coords[..., 0].long().clamp(0, self.W - 1)  # 宽度
        y_coords = norm_coords[..., 1].long().clamp(0, self.H - 1)  # 高度
        
        # 创建批次索引
        batch_indices = torch.arange(batch_size, device=norm_coords.device)[:, None].expand(-1, self.total_nodes)
        
        # 准备通道索引（全部是0）
        channel_indices = torch.zeros_like(batch_indices)
        
        # 准备要赋的值
        values = cluster_idx.float() + 1  # 确保值不为0，0保留为背景
        
        # 使用索引赋值 - 注意正确的顺序：[batch, channel, height(y), width(x)]
        image[batch_indices.flatten(), 
              channel_indices.flatten(),
              y_coords.flatten(), 
              x_coords.flatten()] = values.flatten()
        
        return image


class PatchEmbedding(nn.Module):
    """将图像分割成patch并进行嵌入"""
    def __init__(self, **model_param):
        super().__init__()
        self.patch_size = model_param['patch_size']
        self.in_channels = model_param['in_channels']
        self.embedding_dim = model_param['embedding_dim']
        
        # 使用卷积层实现patching和embedding
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
        # 通过卷积实现patching
        x = self.proj(x)        # [batch_size, embed_dim, H//patch_size, W//patch_size]
        
        # 重排维度
        x = x.flatten(2)        # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (batch_size, num_patches, embed_dim)
        return x


class ScalablePositionalEncoding(nn.Module):
    """可扩展的位置编码，使用MLP生成"""
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
            位置编码 (batch_size, num_patches, embed_dim)
        """
        batch_size, num_patches, _ = x.shape
        
        # 计算大致的H和W，可能不是完全平方数
        H = int(num_patches ** 0.5)
        W = num_patches // H
        if H * W < num_patches:
            W += 1  # 确保H*W >= num_patches
            
        # 生成网格坐标
        grid_h = torch.linspace(0, 1, H, device=x.device)
        grid_w = torch.linspace(0, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        
        # [H*W, 2]
        grid = torch.stack([grid_w, grid_h], dim=-1).reshape(-1, 2)
        
        # 如果H*W > num_patches，截断多余的部分
        grid = grid[:num_patches]
        
        # 通过mlp生成位置编码
        pos_embed = self.mlp(grid)  # [num_patches, embedding_dim]
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pos_embed


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
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
        
        # 使用LayerNorm代替InstanceNorm
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
        x = self.norm1(x)  # 直接应用LayerNorm，不需要转置
        
        # Feed-forward
        residual = x
        x = self.ffn(x)
        x = residual + x
        x = self.norm2(x)  # 直接应用LayerNorm，不需要转置
        
        return x


class VisionTransformer(nn.Module):
    """处理坐标图像的Vision Transformer"""
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