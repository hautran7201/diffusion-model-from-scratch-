import torch 
from torch import nn 
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock 

class VAE_encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch, 3, hieght, width) -> (Batch, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch, 128, hieght, width) -> (Batch, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (Batch, 128, hieght, width) -> (Batch, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (Batch, 128, hieght, width) -> (Batch, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch, 128, hieght/2, width/2) -> (Batch, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),

            # (Batch, 256, hieght/2, width/2) -> (Batch, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch, 256, hieght/2, width/2) -> (Batch, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch, 256, hieght/4, width/4) -> (Batch, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),

            # (Batch, 512, hieght/4, width/4) -> (Batch, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch, 512, hieght/4, width/4) -> (Batch, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (Batch, 512, hieght/8, width/8) -> (Batch, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch, 512, hieght/8, width/8) -> (Batch, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch, 512, hieght/8, width/8) -> (Batch, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch, 512, hieght/8, width/8) -> (Batch, 512, height/8, width/8)
            VAE_AttentionBlock(512),

            # (Batch, 512, hieght/8, width/8) -> (Batch, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch, 512, hieght/8, width/8) -> (Batch, 512, height/8, width/8)
            nn.GroupNorm(32, 512),

            # (Batch, 512, hieght/8, width/8) -> (Batch, 512, height/8, width/8)
            nn.SiLU(),

            # (Batch, 512, hieght/8, width/8) -> (Batch, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch, 512, hieght/8, width/8) -> (Batch, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Chanel, Height, Width)
        # noise: (Batches, OutputChanel, Height, Width)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (Batch, 8, Height / 8, Width / 8) -> 2 tensor (Batch, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=-1)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        variance = log_variance.exp()

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        stdvar = variance.sqrt()

        # Z=N(0, 1) -> X=N(mean, stdvar)
        x = mean + stdvar*x

        # Scale the output by constant
        x *= 0.18215

        return x 