import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocal_size, n_embedd, n_tokens):
        super().__init__()

        self.token_embedding = nn.Embedding(vocal_size, n_embedd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embedd))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embedd: int):
        self.layernorm_1 = nn.LayerNorm(n_embedd)
        self.selfattetion = SelfAttention(n_head, n_embedd)
        self.layernorm_2 = nn.LayerNorm(n_embedd)
        self.linear_1 = nn.Linear(n_embedd, 4 * n_embedd)
        self.linear_2 = nn.Linear(4 * n_embedd, n_embedd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, Seg_len, Dim)

        residual = x

        ## Self Attention
        x = self.layernorm_1(x)
        x = self.selfattetion(x, casual_mask=True)
        x += residual

        residual = x

        ## Feed forward
        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702)

        x = self.linear_2(x)

        x += residual 

        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layer = nn.Module(
            [
                CLIPLayer(12, 768) for i in range(12)
            ]
        )

        self.layernorm = nn.LayerNorm(
            768
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layer:
            state = layer(state)

        output = self.layernorm(state)

        return output