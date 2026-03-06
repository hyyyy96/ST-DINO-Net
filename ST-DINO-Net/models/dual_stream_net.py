import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion import BiDirectionalGatedFusion
from .heads import MultiScaleGEMHead
from .backbone import ResNet18MotionEncoder


class DualStreamCloudNet(nn.Module):
    """
    Dual-Stream Network for Cloud Classification
    Combines DINO (spatial) and ResNet (motion) features
    """

    def __init__(self, dino_model, num_classes, dino_feature_dim, head_type="gem", dropout=0.3):
        super(DualStreamCloudNet, self).__init__()

        # Spatial Stream (DINO)
        self.spatial_backbone = dino_model
        self.spatial_dim = dino_feature_dim

        # Motion Stream (ResNet-18)
        self.motion_backbone = ResNet18MotionEncoder(pretrained=False)
        self.motion_dim = 512

        # Fusion Module
        self.fusion_module = BiDirectionalGatedFusion(
            dim_s=self.spatial_dim,
            dim_m=self.motion_dim,
            num_heads=8,
            dropout=dropout
        )

        # Main Classification Head
        fusion_out_dim = self.spatial_dim
        if head_type == "gem":
            self.head = MultiScaleGEMHead(fusion_out_dim, num_classes, dropout=dropout)
        else:
            self.head = nn.Linear(fusion_out_dim, num_classes)

        # Auxiliary Heads (for deep supervision during training)
        self.aux_head_spatial = nn.Linear(self.spatial_dim, num_classes)
        self.aux_head_motion = nn.Linear(self.motion_dim, num_classes)

    def forward(self, inputs, return_features=False):
        """
        Forward pass

        Args:
            inputs: tuple (rgb_images, flow_images)
            return_features: if True, return features for visualization

        Returns:
            Classification logits or (logits, features)
        """
        x_rgb, x_flow = inputs

        # Spatial features from DINO
        feat_s = self.spatial_backbone(x_rgb)
        if isinstance(feat_s, (tuple, list)):
            feat_s = feat_s[0]
        if isinstance(feat_s, dict):
            feat_s = feat_s.get("x_norm_clstoken", list(feat_s.values())[-1])

        # Flatten spatial features
        if feat_s.ndim == 4:
            feat_s = F.adaptive_avg_pool2d(feat_s, (1, 1)).flatten(1)
        elif feat_s.ndim == 3:
            feat_s = feat_s[:, 0, :]

        # Motion features from ResNet
        feat_m = self.motion_backbone(x_flow)

        # Feature fusion
        feat_fused = self.fusion_module(x_s=feat_s, x_m=feat_m)

        # Classification
        out_main = self.head(feat_fused)

        if return_features:
            return out_main, {
                'spatial': feat_s,
                'motion': feat_m,
                'fused': feat_fused
            }

        return out_main

    def get_intermediate_features(self, inputs):
        """Get intermediate features for analysis"""
        return self.forward(inputs, return_features=True)[1]