import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_dino_backbone(repo_dir, version="dinov3_vith16plus", pretrained_path=None, device='cpu'):
    """
    Build DINO backbone from local repository
    """
    try:
        dino_model = torch.hub.load(repo_dir, version, source='local',
                                    weights=pretrained_path, trust_repo=True)
    except:
        print("Warning: Failed to load DINO from local repo, trying direct load...")
        import sys
        sys.path.append(repo_dir)
        # Alternative loading method
        dino_model = torch.hub.load('facebookresearch/dinov3', version)

    return dino_model.to(device)


def get_dino_feature_dim(dino_model, device='cpu'):
    """
    Get feature dimension from DINO backbone
    """
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        out = dino_model(dummy)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):
            out = list(out.values())[-1]
        if out.ndim == 3:
            dim = out.shape[2]  # For ViT: [B, N, D]
        else:
            dim = out.shape[1]  # For CNN
    return dim


class ResNet18MotionEncoder(nn.Module):
    """Motion encoder using ResNet-18"""

    def __init__(self, pretrained=False):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = 512

    def forward(self, x):
        return self.backbone(x).flatten(1)