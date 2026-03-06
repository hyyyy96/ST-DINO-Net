#!/usr/bin/env python
"""
Evaluation script for ST-DINO-Net
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_stream_net import DualStreamCloudNet
from models.backbone import build_dino_backbone, get_dino_feature_dim
from data.dataset import DualStreamDataset
from utils.metrics import compute_metrics, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ST-DINO-Net')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--rgb_root', type=str, required=True,
                        help='Root directory of RGB test images')
    parser.add_argument('--flow_root', type=str, default=None,
                        help='Root directory of flow test images')
    parser.add_argument('--repo_dir', type=str,
                        default=None,
                        help='DINO repository directory (optional)')
    parser.add_argument('--dino_version', type=str,
                        default="dinov2_vitb14",  # 改为更通用的版本
                        help='DINO version')
    parser.add_argument('--dino_weights', type=str,
                        default=None,
                        help='DINO pretrained weights (optional)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--save_cm', type=str, default='confusion_matrix.png',
                        help='Path to save confusion matrix')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='Number of classes')
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    print("Loading dataset...")
    test_dataset = DualStreamDataset(
        rgb_root=args.rgb_root,
        flow_root=args.flow_root,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    class_names = test_dataset.get_class_names()
    print(f"Test set size: {len(test_dataset)}, Classes: {len(class_names)}")

    # Build model
    print("Building model...")

    # Try to load DINO from torch hub
    print(f"Loading DINO {args.dino_version} from torch hub...")
    try:
        dino_backbone = torch.hub.load('facebookresearch/dinov2', args.dino_version).to(device)
        print("DINO loaded successfully")
    except Exception as e:
        print(f"Failed to load DINO from hub: {e}")
        print("Using simple CNN backbone as fallback...")
        # Fallback to a simple CNN
        from models.backbone import ResNet18MotionEncoder
        dino_backbone = ResNet18MotionEncoder(pretrained=False)
        dino_dim = 512
        print("Using fallback backbone - please install DINOv2 for full functionality")

    # Get feature dimension
    if 'dino_dim' not in locals():
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            out = dino_backbone(dummy)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, dict):
                out = list(out.values())[-1]
            if out.ndim == 3:
                dino_dim = out.shape[2]
            elif out.ndim == 4:
                dino_dim = out.shape[1]
            else:
                dino_dim = out.shape[1] if out.ndim > 1 else out.shape[0]

    # Create model
    model = DualStreamCloudNet(
        dino_model=dino_backbone,
        num_classes=len(class_names),
        dino_feature_dim=dino_dim,
        head_type="gem"
    ).to(device)

    # Load weights
    print(f"Loading weights from {args.weights}...")
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        return

    # Load state dict with appropriate settings
    state_dict = torch.load(args.weights, map_location=device)

    # Handle different state dict formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    try:
        model.load_state_dict(state_dict, strict=True)
        print("Weights loaded successfully (strict mode).")
    except RuntimeError as e:
        print(f"Strict loading failed, trying non-strict: {e}")
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded with non-strict mode.")

    model.eval()

    # Inference
    print("Running inference...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, ((img_rgb, img_flow), labels) in enumerate(test_loader):
            img_rgb = img_rgb.to(device)
            img_flow = img_flow.to(device)
            labels = labels.to(device)

            outputs = model((img_rgb, img_flow))
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, class_names)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1']:.4f}")
    print("\nDetailed Classification Report:")
    print(metrics['report'])

    # Plot confusion matrix
    if args.save_cm:
        plot_confusion_matrix(
            all_labels, all_preds,
            class_names,
            save_path=args.save_cm
        )
        print(f"Confusion matrix saved to: {args.save_cm}")

    return metrics


if __name__ == "__main__":
    main()