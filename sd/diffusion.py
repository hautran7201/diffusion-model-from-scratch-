import sys
sys.path.append("sd")

import torch 
from torch import nn 
from torch.nn import functional as F
from clip import CLIP
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embedd):
        super().__init__()
        self.layer_1 = nn.Linear(n_embedd, 4*n_embedd)
        self.layer_2 = nn.Linear(4*n_embedd, n_embedd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)

        x = self.layer_1(x)

        x = nn.SiLU(x)

        x = self.layer_2(x)

        # (1, 1280)
        return x
    

class Upsample(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, Channel, Height, Width) -> (Batch, Channel, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel:int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channel)
        self.conv_feature = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channel)

        self.groupnorm_merged = nn.GroupNorm(32, out_channel)
        self.conv_merged = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, feature, time):
        # feature: (Batch, In_channel, Height, Width)
        # time: (1, 1280)

        residaul = feature 

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residaul)
    

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embedd: int, d_context: int = 768):
        super().__init__()
        channel = n_head * n_embedd

        self.groupnorm = nn.GroupNorm(32, channel, eps=1e-6)
        self.conv_input = nn.Conv2d(channel, channel, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channel)
        self.attention_1 = SelfAttention(n_head, channel, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channel)
        self.attention_2 = CrossAttention(n_head, channel, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channel)
        self.linear_geglu_1 = nn.Linear(channel, 4 * channel * 2)
        self.linear_geglu_2 = nn.Linear(4 * channel, channel)

        self.conv_output = nn.Conv2d(channel, channel, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Feature, Height, Width)
        # context: (Batch, Seg_len, Dim)

        residual_long = x 

        x = self.groupnorm(x)

        x = self.conv_input(x)

        b, c, h, w = x.shape

        # (Batch, Feature, Height, Width) -> (Batch, Feature, Height * Width)
        x = x.view(b, c, h * w)

        x = x.transpose(-1, -2)

        residual_short = x 

        ## Noramlizztion + Self Attention with skip connection

        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residual_short

        residual_short = x

        ## Noramlizztion + Cross Attention with skip connection

        x = self.layernorm_2(x)
        self.attention_2(x)
        x += residual_short

        residual_short = x

        ## Normalisation + FF with geglu and skip connection

        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residual_short

        # (Batch, Height * Width, Feature) -> (Batch, Feature, Height * Width)
        x = x.transpose(-1, -2)

        x = x.view((b, c, h, w))

        return self.conv_output(x) + residual_long


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([
            # (Batch, 4, Height / 8, Width / 8) -> # (Batch, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (Batch, 320, Height / 8, Width / 8) -> (Batch, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (Batch, 640, Height / 16, Width / 16) -> (Batch, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch, 1280, Height / 32, Width / 32) -> (Batch, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)  
        )

        self.decoder = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        print('start')
        skip_conections = []
        for layer in self.encoder:
            x = layer(x, context, time)
            skip_conections.append(x)
        
        x = self.bottleneck(x)

        for layer in self.decoder:
            skip = skip_conections.pop()
            print("Shape của layer:", x)
            print("Shape của skip:", skip)
            x = torch.cat((x, skip), dim=1)
            x = layer(x, context, time)
        
        return x

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (Batch, 320, Height / 8, Width / 8)

        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        # (Batch, Out_channel, Height / 8, Width / 8)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = Unet()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, lantent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # lantent: (Batch, 4, Height / 8, Width / 8)
        # context: (Batch, Seg_len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(lantent, context, time)

        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)

        return output
