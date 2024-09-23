import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)  # Self-attention
        self.cross_attn = MultiHeadAttention(d_model, n_heads)  # Cross-attention (encoder-decoder attention)
        self.ff = FeedForwardLayer(d_model, d_ff)  # Feedforward layer
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.dropout3 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
        self.residual3 = ResidualConnection()
    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.residual1(x, lambda x: self.dropout1(self.self_attn(x, x, x, tgt_mask)))  # Self-attention block
        x = self.norm1(x)

        # Cross-attention with residual connection and normalization
        x = self.residual2(x, lambda x: self.dropout2(self.cross_attn(x, memory, memory, src_mask)))  # Cross-attention block
        x = self.norm2(x)

        # Feedforward with residual connection and normalization
        x = self.residual3(x, lambda x: self.dropout3(self.ff(x)))  # Feedforward block
        x = self.norm3(x)
        return x