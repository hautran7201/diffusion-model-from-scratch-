import torch 
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention 

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channel):
        self.attention_layer = SelfAttention(1, channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channel, Height, Width)

        residual = x

        b, c, h, w = x.shape

        # (Batch, Channel, Height, Width) -> (Batch, Channel, Height * Width)
        x = x.view(b, c, h*w)

        # (Batch, Channel, Height * Width) -> (Batch, Height * Width, Channel)
        x = x.transpose(-1, -2)

        # (Batch, Height * Width, Channel) -> (Batch, Height * Width, Channel)
        x = self.attention_layer(x)

        # (Batch, Channel, Height * Width) -> (Batch, Channel, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch, Channel, Height * Width) -> (Batch, Channel, Height, Width)
        x = x.view(b, c, h, w)

        x += residual

        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.groupnorm_1 = nn.GroupNorm(32, output_dim)
        self.conv_1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, output_dim)
        self.conv_2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)

        if input_dim == output_dim:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(input_dim, output_dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channel, Height, Width)

        residual = x

        x = self.groupnorm_1(x)

        x = nn.SiLU(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = nn.SiLU(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residual)

class VAE_Decoder(nn.Sequential):
    def __init__(self,):
        super().__init__(
            # (Batch, 4, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # (Batch, 4, Height / 8, Width / 8) -> (Batch, 512, Height / 8, Width / 8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),            

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),             

            VAE_ResidualBlock(512, 512), 

            VAE_ResidualBlock(512, 512), 

            # (Batch, 512, Height / 8, Width / 8) -> (Batch, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 

            # (Batch, 512, Height / 8, Width / 8) -> (Batch, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),             
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 

            # (Batch, 512, Height / 4, Width / 4) -> (Batch, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),  

            VAE_ResidualBlock(512, 256),             
            VAE_ResidualBlock(256, 256), 
            VAE_ResidualBlock(256, 256), 

            # (Batch, 512, Height / 2, Width / 2) -> (Batch, 512, Height, Width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),  

            VAE_ResidualBlock(256, 128),             
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 4, Height, Width)

        x /= 0.18215

        for module in self:
            x = module(x)

        return x