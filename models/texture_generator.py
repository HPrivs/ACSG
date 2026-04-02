from models.unet_parts import *
import torch

class FourierEncoding2D(nn.Module):
    def __init__(self, input_dims, num_freqs, include_input=True, log_sampling=True, device='cuda'):
        super(FourierEncoding2D, self).__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.periodic_fns = [torch.sin, torch.cos]
        self.device = device

        self.freq_bands = None
        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0.0, num_freqs - 1, steps=num_freqs, device=self.device)
        else:
            self.freq_bands = torch.linspace(2.0 ** 0.0, 2. ** (num_freqs - 1), steps=num_freqs, device=self.device)

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dims
        self.out_dim += self.input_dims * self.num_freqs * len(self.periodic_fns)

    def forward(self, coords):
        encoding = [coords] if self.include_input else []
        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                feat = p_fn(coords * freq)
                encoding.append(feat)

        return torch.cat(encoding, -1)

class StructurePerturbation(nn.Module):
    def __init__(self, in_channels, n_channels=64, out_channels=3, bilinear=True):
        super(StructurePerturbation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, n_channels)

        self.down1 = Down(n_channels, n_channels * 2)
        self.down2 = Down(n_channels * 2, n_channels * 4)
        self.down3 = Down(n_channels * 4, n_channels * 8)

        factor = 2 if bilinear else 1
        self.bottleneck = DoubleConv(n_channels * 8, n_channels * 8 // factor)

        self.up1 = Up(n_channels * 8, n_channels * 4 // factor, bilinear)
        self.up2 = Up(n_channels * 4, n_channels * 2 // factor, bilinear)
        self.up3 = Up(n_channels * 2, n_channels, bilinear)

        self.outc = OutConv(n_channels, out_channels)

    def forward(self, x_pos_encoded):
        # Encoder
        x1 = self.inc(x_pos_encoded)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        # Decoder
        x = self.up1(x5, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        output = torch.sigmoid(logits)
        return output


class AdversarialTextureGenerator(nn.Module):
    def __init__(self, shape=(512, 512), device='cuda', latent_dim=32, num_heads=4, num_freqs=8):
        super().__init__()
        self.device = device
        self.shape = shape
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_freqs = num_freqs

        # Fourier 编码
        self.cartesian_fourier_encoder = FourierEncoding2D(input_dims=4, num_freqs=self.num_freqs).to(device)
        self.encoding_proj = nn.Conv2d(self.cartesian_fourier_encoder.out_dim, 128, kernel_size=1).to(device)
        self.perturber = StructurePerturbation(in_channels=128, n_channels=128, out_channels=3).to(device)

        # 坐标
        y_coords = torch.linspace(-1.0, 1.0, shape[0], device=device)
        x_coords = torch.linspace(-1.0, 1.0, shape[1], device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer("yy_base", yy)
        self.register_buffer("xx_base", xx)

    def forward(self):
        cartesian_coords = torch.stack([self.yy_base, self.xx_base], dim=-1)
        coords_times_pi = cartesian_coords * torch.pi
        torus_coords = torch.cat([torch.cos(coords_times_pi), torch.sin(coords_times_pi)], dim=-1)

        encoded_cartesian = self.cartesian_fourier_encoder(torus_coords)
        features = self.encoding_proj(encoded_cartesian.permute(2, 0, 1).unsqueeze(0))  # [1, 128, H, W]

        perturb_noise = self.perturber(features)  # [1, 3, H, W]

        return perturb_noise

# if __name__ == "__main__":
    # texture_generator = AdversarialTextureGenerator(shape=(128, 128)).to(device='cuda')
    # perturb = texture_generator()
    #
    # perturb_img = Image.fromarray((perturb * 255).squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    # perturb_img.save('../debug/perturb.png')
