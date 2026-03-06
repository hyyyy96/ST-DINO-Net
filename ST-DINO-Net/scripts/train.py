#!/usr/bin/env python
"""
Training script for ST-DINO-Net
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_stream_net import DualStreamCloudNet
from models.backbone import build_dino_backbone, get_dino_feature_dim
from data.dataset import DualStreamDataset
from utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train ST-DINO-Net')
    parser.add_argument('--train_rgb', type=str, required=True,
                        help='Root directory of RGB training images')
    parser.add_argument('--train_flow', type=str, default=None,
                        help='Root directory of flow training images')
    parser.add_argument('--val_rgb', type=str, required=True,
                        help='Root directory of RGB validation images')
    parser.add_argument('--val_flow', type=str, default=None,
                        help='Root directory of flow validation images')
    parser.add_argument('--dino_version', type=str,
                        default="dinov2_vitb14",
                        help='DINO version (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)')
    parser.add_argument('--unfreeze_layers', type=int, default=2,
                        help='Number of DINO layers to unfreeze (0=frozen, -1=all)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--lambda_aux', type=float, default=0.3,
                        help='Weight for auxiliary losses')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval (batches)')
    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, device, lambda_aux, epoch, log_interval):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} Training')
    for batch_idx, ((img_rgb, img_flow), labels) in enumerate(pbar):
        img_rgb = img_rgb.to(device)
        img_flow = img_flow.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model((img_rgb, img_flow))

        # Main loss
        loss_main = criterion(outputs, labels)

        # Auxiliary losses
        loss_aux_s = 0
        loss_aux_m = 0
        if hasattr(model, 'aux_head_spatial') and hasattr(model, 'aux_head_motion'):
            # Get intermediate features
            with torch.no_grad():
                features = model.get_intermediate_features((img_rgb, img_flow))
            out_aux_s = model.aux_head_spatial(features['spatial'])
            out_aux_m = model.aux_head_motion(features['motion'])
            loss_aux_s = criterion(out_aux_s, labels)
            loss_aux_m = criterion(out_aux_m, labels)

        # Total loss
        loss = loss_main + lambda_aux * (loss_aux_s + loss_aux_m)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss / (batch_idx + 1):.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

        # Log occasionally
        if (batch_idx + 1) % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'\nEpoch {epoch}, Batch {batch_idx + 1}/{len(loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%, LR: {current_lr:.2e}')

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for (img_rgb, img_flow), labels in tqdm(loader, desc='Validating'):
            img_rgb = img_rgb.to(device)
            img_flow = img_flow.to(device)
            labels = labels.to(device)

            outputs = model((img_rgb, img_flow))
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)

    return total_loss / len(loader), metrics['accuracy'] * 100, metrics


def main():
    args = parse_args()

    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"stdinonet_{timestamp}"
    save_dir = os.path.join(args.save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("Loading datasets...")
    train_dataset = DualStreamDataset(
        rgb_root=args.train_rgb,
        flow_root=args.train_flow,
        transform=train_transform
    )

    val_dataset = DualStreamDataset(
        rgb_root=args.val_rgb,
        flow_root=args.val_flow,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    class_names = train_dataset.get_class_names()
    num_classes = len(class_names)
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    # Build model
    print("Building model...")
    print(f"Loading DINO {args.dino_version} from torch hub...")

    try:
        dino_backbone = torch.hub.load('facebookresearch/dinov2', args.dino_version).to(device)
        print("DINO loaded successfully")
    except Exception as e:
        print(f"Failed to load DINO: {e}")
        print("Please install DINOv2: pip install dinov2")
        return

    # Get feature dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        out = dino_backbone(dummy)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):
            out = list(out.values())[-1]
        if out.ndim == 3:
            dino_dim = out.shape[2]
        else:
            dino_dim = out.shape[1]

    print(f"DINO feature dimension: {dino_dim}")

    # Freeze/unfreeze DINO layers
    if hasattr(dino_backbone, 'blocks'):
        total_blocks = len(dino_backbone.blocks)
        if args.unfreeze_layers == 0:
            for param in dino_backbone.parameters():
                param.requires_grad = False
            print("DINO backbone frozen completely")
        elif args.unfreeze_layers > 0:
            # Freeze all first
            for param in dino_backbone.parameters():
                param.requires_grad = False
            # Unfreeze last N blocks
            n_to_unfreeze = min(args.unfreeze_layers, total_blocks)
            for i, block in enumerate(dino_backbone.blocks[-n_to_unfreeze:]):
                for param in block.parameters():
                    param.requires_grad = True
            print(f"Unfroze last {n_to_unfreeze}/{total_blocks} transformer blocks")
        elif args.unfreeze_layers == -1:
            print("Training all DINO layers (full fine-tuning)")

    # Create model
    model = DualStreamCloudNet(
        dino_model=dino_backbone,
        num_classes=num_classes,
        dino_feature_dim=dino_dim,
        head_type="gem",
        dropout=0.3
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100. * trainable_params / total_params:.2f}%)")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint.get('epoch', 0)
            best_acc = checkpoint.get('best_acc', 0)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
        else:
            print(f"No checkpoint found at {args.resume}")

    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 60}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            args.lambda_aux, epoch + 1, args.log_interval
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update scheduler
        scheduler.step()

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, best_path)
            print(f"  🏆 New best model saved! Accuracy: {val_acc:.2f}% -> {best_path}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Save latest model
        latest_path = os.path.join(save_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args)
        }, latest_path)

    print(f"\n{'=' * 60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {save_dir}")
    print(f"{'=' * 60}")

    # Save final model
    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")

    return best_acc


if __name__ == "__main__":
    main()