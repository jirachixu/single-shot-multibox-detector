import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F_t
from tqdm import tqdm

from model.model import SSDModel

# use gpu if available for faster computation (ran on cuda 13.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    '''
    Custom collate function for the DataLoader. Since each image can have a different 
    number of bounding boxes, we can't just stack the target tensors into a single batch tensor. 
    Instead, we zip them into a tuple of images and a tuple of targets.
    
    Args:
        **batch (list)**: List of tuples (image, target) from the dataset.
        
    Returns:
        **tuple**: A tuple containing (images, targets).
    '''
    return tuple(zip(*batch))

def get_coco_category_map(coco):
    '''
    Creates a mapping from the original 80 COCO category IDs (which are non-contiguous, 
    often ranging from 1 to 90) to a contiguous sequence of IDs from 0 to 79.
    
    Args:
        **coco**: The COCO API object.
        
    Returns:
        **tuple**: A tuple containing two dictionaries (coco_id_to_contiguous, contiguous_to_coco_id).
    '''
    cats = coco.cats
    # sort by original category id to ensure consistent mapping
    coco_id_to_contiguous = {coco_id: i for i, coco_id in enumerate(sorted(cats.keys()))}
    contiguous_to_coco_id = {i: coco_id for coco_id, i in coco_id_to_contiguous.items()}
    return coco_id_to_contiguous, contiguous_to_coco_id

class COCODatasetWrapper(Dataset):
    '''
    Wrapper around torchvision's CocoDetection dataset. This wrapper is responsible for 
    resizing images to 300x300 (required by our SSD architecture) and normalizing the 
    bounding box coordinates to be between 0 and 1, which is expected by our loss function.
    
    Args:
        **root (str)**: Root directory where images are downloaded to.
        **annFile (str)**: Path to json annotation file.
    '''
    def __init__(self, root: str, annFile: str):
        super().__init__()
        self.coco_ds = CocoDetection(root=root, annFile=annFile)
        self.coco_id_to_contiguous, _ = get_coco_category_map(self.coco_ds.coco)
        
        # filter out images with no annotations, as they will break the loss calculation
        self.valid_indices = []
        for i in range(len(self.coco_ds)):
            _, target = self.coco_ds[i]
            if len(target) > 0:
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        '''
        Retrieves the image and bounding boxes, resizes the image, and normalizes the boxes.
        '''
        real_idx = self.valid_indices[idx]
        img, target = self.coco_ds[real_idx]
        
        orig_w, orig_h = img.size
        
        # convert PIL image to tensor and resize to 300x300
        img_t = F_t.to_tensor(img)
        img_t = F_t.resize(img_t, [300, 300], antialias=True)
        
        boxes = []
        labels = []
        
        for obj in target:
            # COCO bounding boxes are in [x_min, y_min, width, height] format
            x, y, w, h = obj['bbox']
            
            # calculate x_max and y_max
            x_max = x + w
            y_max = y + h
            
            # normalize the coordinates to [0, 1] based on original image size
            # this makes the bounding boxes scale-invariant, aligning with our anchor boxes
            x_min_norm = x / orig_w
            y_min_norm = y / orig_h
            x_max_norm = x_max / orig_w
            y_max_norm = y_max / orig_h
            
            # clamp coordinates just to be safe
            x_min_norm = max(0.0, min(x_min_norm, 1.0))
            y_min_norm = max(0.0, min(y_min_norm, 1.0))
            x_max_norm = max(0.0, min(x_max_norm, 1.0))
            y_max_norm = max(0.0, min(y_max_norm, 1.0))
            
            boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
            
            # map the COCO category to our contiguous 0-79 mapping
            contig_id = self.coco_id_to_contiguous.get(obj['category_id'], -1)
            labels.append(contig_id)
            
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        
        return img_t, {'boxes': boxes_tensor, 'labels': labels_tensor}

def main(args):
    print(f"Using device: {device}")
    print(f"Loading COCO dataset from {args.img_dir}...")
    try:
        dataset = COCODatasetWrapper(root=args.img_dir, annFile=args.ann_file)
        print(f"Loaded {len(dataset)} valid images with annotations.")
    except ImportError:
        print("Error: pycocotools is not installed. Please install it using 'pip install pycocotools'.")
        return
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        collate_fn=collate_fn
    )

    print("Initializing SSD Model...")
    # initialize model with the number of contiguous classes (usually 80 for COCO)
    model = SSDModel(n_classes=len(dataset.coco_id_to_contiguous))
    
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded starting weights from {args.weights}")
    else:
        print("No pre-trained weights provided. Training from scratch (using ResNet50 pretrained backbone).")
    
    model.to(device)
    model.train()

    # using AdamW optimizer as it generally converges faster than standard SGD
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    os.makedirs(args.save_dir, exist_ok=True)

    print("Starting training loop...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        # tqdm progress bar for the batches in the epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # images is a tuple of tensors, we stack them into a single batch tensor
            images = torch.stack(images).to(device)
            
            # extract boxes and labels into lists of tensors
            # we keep them as lists because calculate_loss handles them image-by-image
            true_boxes = [t['boxes'].to(device) for t in targets]
            true_labels = [t['labels'].to(device) for t in targets]
            
            optimizer.zero_grad()
            
            # forward pass
            predicted_offsets, predicted_class_scores = model(images)
            
            # calculate loss
            loss = model.calculate_loss(predicted_offsets, predicted_class_scores, true_boxes, true_labels)
            
            # backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # update progress bar with current loss
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # save a checkpoint at the end of every epoch
        checkpoint_path = os.path.join(args.save_dir, f"ssd_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SSD Model on COCO Dataset")
    parser.add_argument('--img_dir', type=str, required=True, help='Path to COCO training images directory (e.g., train2017)')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to COCO training annotations file (e.g., instances_train2017.json)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--weights', type=str, default=None, help='Path to pre-trained model weights (.pth) to resume training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    
    args = parser.parse_args()
    main(args)
