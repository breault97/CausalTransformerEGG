import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.use_res = in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        if self.use_res:
            out = out + x
        return out


class ConvFrontend(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim else int(in_dim)
        layers = int(layers)
        blocks = []
        for i in range(max(layers, 1)):
            in_ch = in_dim if i == 0 else hidden_dim
            k = 3 if i % 2 == 0 else 5
            blocks.append(ConvBlock(in_ch, hidden_dim, k, dilation=1, dropout=dropout))
        self.net = nn.Sequential(*blocks)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class TCNFrontend(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim else int(in_dim)
        layers = int(layers)
        blocks = []
        for i in range(max(layers, 1)):
            in_ch = in_dim if i == 0 else hidden_dim
            dilation = 2 ** i
            blocks.append(ConvBlock(in_ch, hidden_dim, kernel_size=3, dilation=dilation, dropout=dropout))
        self.net = nn.Sequential(*blocks)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


def build_eeg_frontend(frontend_type: str, in_dim: int, hidden_dim: int, layers: int, dropout: float) -> nn.Module:
    ftype = str(frontend_type).lower() if frontend_type is not None else "conv"
    if ftype in ("tcn", "dilated", "dilated_conv"):
        return TCNFrontend(in_dim, hidden_dim, layers, dropout)
    return ConvFrontend(in_dim, hidden_dim, layers, dropout)
