import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized Mean Pooling"""

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class MultiScaleGEMHead(nn.Module):
    """
    Multi-Scale Generalized Mean Pooling Head
    Combines multiple pooling strategies for enhanced feature representation
    """

    def __init__(self, in_dim, num_classes, dropout=0.3):
        super(MultiScaleGEMHead, self).__init__()
        self.gem1 = GeM(p=1.0)
        self.gem3 = GeM(p=3.0)
        self.gem6 = GeM(p=6.0)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.attn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_dim // 4, 1, kernel_size=1),
        )
        fusion_dim = in_dim * 5
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)

        # Multi-scale pooling
        g1 = self.gem1(x)
        g3 = self.gem3(x)
        g6 = self.gem6(x)
        m = self.maxpool(x)

        # Attention pooling
        a = self.attn(x)
        a = torch.softmax(a.view(a.size(0), -1), dim=1).view_as(a)
        attn = (x * a).sum(dim=[2, 3], keepdim=True)

        # Concatenate all features
        out = torch.cat([g1, g3, g6, m, attn], dim=1)
        return self.fc(out.flatten(1))