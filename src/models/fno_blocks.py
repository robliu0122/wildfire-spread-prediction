"""Self-contained Fourier Neural Operator blocks (no `neuraloperator` dependency).

SpectralConv2d: 2D spectral convolution (Li et al. 2021, canonical two-corner form).
FNOBlock: GELU(SpectralConv2d(x) + Conv1x1(x)) — spectral mixing + pointwise bypass.

Used to replace the local-convolution spatial mixing at a chosen ResNet18-UNet stage
with global spectral mixing. Mode counts are capped to what the spatial size allows,
so the same block works at a 4x4 bottleneck or a higher-resolution (e.g. 16x16) stage.
"""
import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_h: int, modes_w: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h          # kept Fourier modes along height (each corner)
        self.modes_w = modes_w          # kept Fourier modes along width (rfft axis)
        scale = 1.0 / (in_channels * out_channels)
        # Two weight tensors: low-positive and low-negative height frequencies.
        self.weight1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_h, modes_w, dtype=torch.cfloat))
        self.weight2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_h, modes_w, dtype=torch.cfloat))

    @staticmethod
    def _compl_mul(x, w):
        # (B, Cin, H, W) x (Cin, Cout, H, W) -> (B, Cout, H, W), complex
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        in_dtype = x.dtype
        x = x.float()                                   # rFFT not supported in half precision
        B, C, H, W = x.shape
        mh = min(self.modes_h, H // 2)                  # avoid top/bottom corner overlap
        mw = min(self.modes_w, W // 2 + 1)              # rfft last dim has W//2 + 1 bins

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mh, :mw] = self._compl_mul(x_ft[:, :, :mh, :mw],
                                                 self.weight1[:, :, :mh, :mw])
        if mh > 0:
            out_ft[:, :, -mh:, :mw] = self._compl_mul(x_ft[:, :, -mh:, :mw],
                                                      self.weight2[:, :, :mh, :mw])
        out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return out.to(in_dtype)


class FNOBlock(nn.Module):
    """One FNO layer: global spectral mixing + pointwise bypass, then GELU."""
    def __init__(self, channels: int, modes_h: int, modes_w: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_h, modes_w)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.bypass(x))
