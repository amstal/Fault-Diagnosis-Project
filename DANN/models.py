import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, List

categories: List[str] = [
    'Healthy',
    'Motor_1_Stuck', 'Motor_1_Steady_state_error',
    'Motor_2_Stuck', 'Motor_2_Steady_state_error',
    'Motor_3_Stuck', 'Motor_3_Steady_state_error',
    'Motor_4_Stuck', 'Motor_4_Steady_state_error'
]
cat2idx = {c: i for i, c in enumerate(categories)}

class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def grad_reverse(x, alpha: float = 1.0):
    return _GradReverse.apply(x, alpha)

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, stride, pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, 1, pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.short = (nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm1d(out_ch)) if in_ch != out_ch or stride != 1 else nn.Identity())
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.short(x)
        return self.relu(out)

class FeatureCNN(nn.Module):
    def __init__(self, in_channels: int = 6, feat_dim: int = 256):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, 64, 7, padding=3, bias=False)
        self.layer = nn.Sequential(
            BasicBlock1D(64, 128, stride=2),
            BasicBlock1D(128, 128),
            BasicBlock1D(128, 256, stride=2),
            BasicBlock1D(256, 256),
        )
        self.se_fc1 = nn.Linear(256, 256 // 16)
        self.se_fc2 = nn.Linear(256 // 16, 256)
        self.pool   = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.stem(x))
        x = self.layer(x)
        w = self.pool(x).squeeze(-1)
        s = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(w)))).unsqueeze(-1)
        x = x * s
        return self.pool(x).squeeze(-1)

class LabelClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, f):
        return self.fc(f)

class DomainClassifier(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
    def forward(self, f):
        return self.net(f)

class DANN(nn.Module):
    def __init__(self, in_channels: int = 6, num_classes: int = 9, feat_dim: int = 256):
        super().__init__()
        self.FE = FeatureCNN(in_channels, feat_dim)
        self.LC = LabelClassifier(feat_dim, num_classes)
        self.DC = DomainClassifier(feat_dim)

    def forward(self, x, alpha: float = 0.0):
        f = self.FE(x)
        y = self.LC(f)
        d = self.DC(grad_reverse(f, alpha))
        return y, d

def _read_single_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=',')

def load_csv_from_dir(root: str,
                      mean: torch.Tensor = None,
                      std : torch.Tensor = None,
                      device: str | torch.device = 'cpu'
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for label in categories:
        folder = os.path.join(root, label)
        for f in glob.glob(os.path.join(folder, '*.csv')):
            xs.append(_read_single_csv(f))
            ys.append(cat2idx[label])
    if not xs:
        raise RuntimeError(f'Pas de csv trouvés dans {root}')

    X = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
    residual = X[:, :, :3] - X[:, :, 3:6]
    X[:, :, 3:6] = residual

    if mean is None or std is None:
        mean = X.mean(dim=(0, 1), keepdim=True)
        std  = X.std (dim=(0, 1), keepdim=True).clamp_min(1e-8)
    X = (X - mean) / std

    y = torch.tensor(ys, dtype=torch.long, device=device)
    return X, y, mean, std
