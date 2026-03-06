#!/usr/bin/env python
"""
Quick test script for ST-DINO-Net
This script demonstrates a minimal working example with synthetic data
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_stream_net import DualStreamCloudNet


class DummyDINO(nn.Module):
    """Dummy DINO model for quick testing"""

    def __init__(self, out_dim=1024):
        super().__init__()
        self.conv = nn.Conv2d(3, out_dim, kernel_size=1)

    def forward(self, x):
        return self.conv(x).mean(dim=[2, 3], keepdim=True)


class DummyDataset(Dataset):
    """Dummy dataset for quick testing"""

    def __init__(self, num_samples=10, num_classes=7, img_size=224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random RGB image
        img_rgb = torch.randn(3, self.img_size, self.img_size)
        # Random flow image
        img_flow = torch.randn(3, self.img_size, self.img_size)
        # Random label
        label = np.random.randint(0, self.num_classes)
        return (img_rgb, img_flow), label


def quick_test():
    """Run a quick test with dummy data"""
    print("=" * 60)
    print("ST-DINO-Net Quick Test")
    print("=" * 60)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    num_classes = 7
    batch_size = 4
    dino_dim = 1024

    # Create dummy DINO backbone
    print("\n1. Creating model...")
    dino_backbone = DummyDINO(out_dim=dino_dim).to(device)

    # Create full model
    model = DualStreamCloudNet(
        dino_model=dino_backbone,
        num_classes=num_classes,
        dino_feature_dim=dino_dim,
        head_type="gem",
        dropout=0.3
    ).to(device)

    print(f"   Model created successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy dataset
    print("\n2. Creating dummy dataset...")
    dataset = DummyDataset(num_samples=16, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"   Dataset size: {len(dataset)} samples")

    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        for batch_idx, ((img_rgb, img_flow), labels) in enumerate(dataloader):
            img_rgb = img_rgb.to(device)
            img_flow = img_flow.to(device)

            outputs = model((img_rgb, img_flow))
            _, preds = torch.max(outputs, 1)

            print(f"   Batch {batch_idx + 1}:")
            print(f"     Input shapes: RGB {img_rgb.shape}, Flow {img_flow.shape}")
            print(f"     Output shape: {outputs.shape}")
            print(f"     Predictions: {preds.cpu().numpy()}")

            if batch_idx >= 1:  # Just test 2 batches
                break

    print("\n4. Testing feature extraction...")
    with torch.no_grad():
        features = model.get_intermediate_features(
            (img_rgb[:1], img_flow[:1])
        )
        print(f"   Spatial feature shape: {features['spatial'].shape}")
        print(f"   Motion feature shape: {features['motion'].shape}")
        print(f"   Fused feature shape: {features['fused'].shape}")

    print("\n" + "=" * 60)
    print("✅ Quick test completed successfully!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    quick_test()