import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class DualStreamDataset(Dataset):
    """
    Dataset for dual-stream input (RGB + Optical Flow)
    """

    def __init__(self, rgb_root, flow_root=None, transform=None, return_path=False):
        """
        Args:
            rgb_root: Root directory of RGB images
            flow_root: Root directory of optical flow images (optional)
            transform: Transformations to apply
            return_path: Whether to return image paths
        """
        self.rgb_root = rgb_root
        self.flow_root = flow_root
        self.transform = transform
        self.return_path = return_path

        if not os.path.exists(rgb_root):
            raise ValueError(f"RGB root not found: {rgb_root}")

        # Get class names from directory structure
        self.classes = sorted([d.name for d in os.scandir(rgb_root) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Collect all samples
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(rgb_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            images = [f.name for f in os.scandir(cls_dir)
                      if f.is_file() and f.name.lower().endswith(('.jpg', '.png', '.jpeg'))]

            for img_name in images:
                self.samples.append((cls_name, img_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls_name, img_name = self.samples[idx]

        # Load RGB image
        rgb_path = os.path.join(self.rgb_root, cls_name, img_name)
        img_rgb = Image.open(rgb_path).convert('RGB')

        # Load or create dummy flow image
        if self.flow_root:
            flow_path = os.path.join(self.flow_root, cls_name, img_name)
            if os.path.exists(flow_path):
                img_flow = Image.open(flow_path).convert('RGB')
            else:
                # Create black image if flow not available
                img_flow = Image.new('RGB', img_rgb.size, (0, 0, 0))
        else:
            img_flow = Image.new('RGB', img_rgb.size, (0, 0, 0))

        # Apply transformations
        if self.transform:
            img_rgb_tensor = self.transform(img_rgb)
            img_flow_tensor = self.transform(img_flow)
        else:
            to_tensor = transforms.ToTensor()
            img_rgb_tensor = to_tensor(img_rgb)
            img_flow_tensor = to_tensor(img_flow)

        label = self.class_to_idx[cls_name]

        if self.return_path:
            return (img_rgb_tensor, img_flow_tensor), label, rgb_path

        return (img_rgb_tensor, img_flow_tensor), label

    def get_class_names(self):
        return self.classes