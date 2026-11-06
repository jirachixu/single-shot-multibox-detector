import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F_t
from tqdm import tqdm

from model.model import SSDModel

# use gpu if available for faster computation (ran on cuda 13.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard ImageNet normalization statistics required by pretrained ResNet50
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def collate_fn(batch):
    '''
    Custom collate function for the DataLoader. Since each image can have a different 
    number of bounding boxes, we can't just stack the target tensors into a single batch tensor. 
    Instead, we zip them into a tuple of images and a tuple of targets.
    '''
    return tuple(zip(*batch))


def get_coco_category_map(coco):
    '''
    Creates a mapping from the original 80 COCO category IDs (which are non-contiguous, 
    often ranging from 1 to 90) to a contiguous sequence of IDs from 0 to 79.
    '''
    cats = coco.cats
    coco_id_to_contiguous = {coco_id: i for i, coco_id in enumerate(sorted(cats.keys()))}
    contiguous_to_coco_id = {i: coco_id for coco_id, i in coco_id_to_contiguous.items()}
    return coco_id_to_contiguous, contiguous_to_coco_id


class COCODatasetWrapper(Dataset):
    '''
    Wrapper around torchvision's CocoDetection dataset. Resizes images to 300x300,
    applies ImageNet normalization, and normalizes bounding box coordinates to [0, 1].
    '''
    def __init__(self, root: str, annFile: str):
        super().__init__()
        self.coco_ds = CocoDetection(root=root, annFile=annFile)
        self.coco_id_to_contiguous, _ = get_coco_category_map(self.coco_ds.coco)
        
        # Fast in-memory filtering without loading JPEGs from disk.
        # Keeps only images that have at least one valid, non-degenerate bounding box.
        self.valid_indices = []
        coco_api = self.coco_ds.coco
        for i, img_id in enumerate(self.coco_ds.ids):
            ann_ids = coco_api.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = coco_api.loadAnns(ann_ids)
            has_valid_box = any(
                ann['bbox'][2] > 1.0 and ann['bbox'][3] > 1.0
                for ann in anns
            )
            if has_valid_box:
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        real_idx = self.valid_indices[idx]
        img, target = self.coco_ds[real_idx]
        
        orig_w, orig_h = img.size
        
        # Convert PIL image to tensor, resize to 300x300, and normalize for ResNet50
        img_t = F_t.to_tensor(img)
        img_t = F_t.resize(img_t, [300, 300], antialias=True)
        img_t = F_t.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        boxes = []
        labels = []
        
        for obj in target:
            # Skip crowd regions if any slipped through
            if obj.get('iscrowd', 0) == 1:
                continue

            # COCO bounding boxes are in [x_min, y_min, width, height] format
            x, y, w, h = obj['bbox']
            
            # Skip degenerate boxes with zero/tiny width or height to prevent log(0) = -inf in loss
            if w <= 1.0 or h <= 1.0:
                continue
            
            x_max = x + w
            y_max = y + h
            
            # Normalize coordinates to [0, 1] based on original image size
            x_min_norm = max(0.0, min(x / orig_w, 1.0))
            y_min_norm = max(0.0, min(y / orig_h, 1.0))
            x_max_norm = max(0.0, min(x_max / orig_w, 1.0))
            y_max_norm = max(0.0, min(y_max / orig_h, 1.0))
            
            # Ensure clamping didn't collapse the box along image borders
            if (x_max_norm - x_min_norm) <= 1e-4 or (y_max_norm - y_min_norm) <= 1e-4:
                continue
            
            contig_id = self.coco_id_to_contiguous.get(obj['category_id'], -1)
            if contig_id == -1:
                continue
                
            boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
            labels.append(contig_id)
            
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)
        
        return img_t, {'boxes': boxes_tensor, 'labels': labels_tensor}


def main(args):
    print(f"Using device: {device}")
    print(f"Loading COCO dataset from {args.img_dir}...")
    try:
        full_dataset = COCODatasetWrapper(root=args.img_dir, annFile=args.ann_file)
        print(f"Loaded {len(full_dataset)} valid images with annotations.")
    except ImportError:
        print("Error: pycocotools is not installed. Please install it using 'pip install pycocotools'.")
        return
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    n_classes = len(full_dataset.coco_id_to_contiguous)

    # Optional: Slice dataset for quick sanity checks (e.g., --subset 4)
    if args.subset is not None and args.subset > 0:
        subset_size = min(args.subset, len(full_dataset))
        dataset = Subset(full_dataset, range(subset_size))
        print(f"Using a subset of {subset_size} images for training/sanity check.")
    else:
        dataset = full_dataset

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn
    )

    print("Initializing SSD Model...")
    model = SSDModel(n_classes=n_classes)
    
    if args.weights and os.path.exists(args.weights):
        # strict=False prevents errors if old checkpoints saved 'anchor_boxes' buffer
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
        print(f"Loaded starting weights from {args.weights}")
    else:
        print("No pre-trained weights provided. Training from scratch (using ResNet50 pretrained backbone).")
    
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Automatic Mixed Precision (AMP) speeds up GPU training ~2x and cuts VRAM usage
    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    print("Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = torch.stack(images).to(device, non_blocking=True)
            
            true_boxes = [t['boxes'].to(device, non_blocking=True) for t in targets]
            true_labels = [t['labels'].to(device, non_blocking=True) for t in targets]
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                predicted_offsets, predicted_class_scores = model(images)
                loss = model.calculate_loss(
                    predicted_offsets,
                    predicted_class_scores,
                    true_boxes,
                    true_labels
                )
            
            scaler.scale(loss).backward()
            # Gradient clipping prevents exploding gradients early in SSD training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save rolling latest checkpoint (prevents filling up disk with 50 separate files)
        latest_path = os.path.join(args.save_dir, "ssd_latest.pth")
        torch.save(model.state_dict(), latest_path)
        
        # Save best checkpoint whenever average loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.save_dir, "ssd_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best loss ({best_loss:.4f})! Saved checkpoint to {best_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SSD Model on COCO Dataset")
    parser.add_argument('--img_dir', type=str, required=True, help='Path to COCO images directory (e.g., train2017 or val2017)')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to COCO annotations file (e.g., instances_train2017.json)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--weights', type=str, default=None, help='Path to pre-trained model weights (.pth) to resume training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--subset', type=int, default=None, help='Train on only N images (e.g., --subset 4 for sanity check)')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (FP16) on CUDA for 2x faster training')
    
    args = parser.parse_args()
    main(args)
