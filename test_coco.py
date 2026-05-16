import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F_t
from tqdm import tqdm

from model.model import SSDModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    return tuple(zip(*batch))

def get_coco_category_map(coco):
    """Creates a mapping from COCO category id to contiguous id (0-79)."""
    cats = coco.cats
    coco_id_to_contiguous = {coco_id: i for i, coco_id in enumerate(sorted(cats.keys()))}
    contiguous_to_coco_id = {i: coco_id for coco_id, i in coco_id_to_contiguous.items()}
    return coco_id_to_contiguous, contiguous_to_coco_id

def main(args):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError:
        print("Error: torchmetrics is not installed. Please install it using 'pip install torchmetrics'.")
        return

    print(f"Loading COCO dataset from {args.img_dir}...")
    try:
        # Note: torchvision.datasets.CocoDetection requires pycocotools to be installed
        dataset = CocoDetection(root=args.img_dir, annFile=args.ann_file)
    except ImportError:
        print("Error: pycocotools is not installed. Please install it using 'pip install pycocotools'.")
        return
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    coco_id_to_contiguous, _ = get_coco_category_map(dataset.coco)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    print("Loading SSD Model...")
    model = SSDModel(n_classes=len(coco_id_to_contiguous))
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        print("Warning: No weights provided or found. Evaluating with random weights.")
    
    model.to(device)
    model.eval()

    # If backend='pycocotools' is specified, it runs faster but requires pycocotools.
    # Since torchvision's CocoDetection already requires pycocotools, we can safely try to use it.
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', backend='pycocotools')

    print("Starting evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            # Prepare images: SSD paper uses 300x300 inputs
            image_tensors = []
            original_sizes = []
            for img in images:
                w, h = img.size
                original_sizes.append((w, h))
                img_t = F_t.to_tensor(img)
                img_t = F_t.resize(img_t, [300, 300], antialias=True)
                image_tensors.append(img_t)
            
            image_tensors = torch.stack(image_tensors).to(device)
            
            # Predict
            predictions = model.predict(image_tensors, iou_threshold=args.iou_threshold, confidence_threshold=args.conf_threshold)
            
            # Format predictions and targets for MeanAveragePrecision
            preds_for_metric = []
            targets_for_metric = []
            
            for i, (pred, target, orig_size) in enumerate(zip(predictions, targets, original_sizes)):
                orig_w, orig_h = orig_size
                
                # Model predictions are normalized [0, 1] xyxy
                pred_boxes = pred['boxes'].clone()
                pred_boxes[:, 0] *= orig_w
                pred_boxes[:, 2] *= orig_w
                pred_boxes[:, 1] *= orig_h
                pred_boxes[:, 3] *= orig_h
                
                # The model outputs contiguous labels 0-79.
                preds_for_metric.append({
                    'boxes': pred_boxes.cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
                
                target_boxes = []
                target_labels = []
                for obj in target:
                    # COCO bbox is [xmin, ymin, width, height]
                    x, y, w, h = obj['bbox']
                    target_boxes.append([x, y, x + w, y + h])
                    # Map COCO category id to contiguous (0-79)
                    contig_id = coco_id_to_contiguous.get(obj['category_id'], -1)
                    target_labels.append(contig_id)
                
                if len(target_boxes) > 0:
                    targets_for_metric.append({
                        'boxes': torch.tensor(target_boxes, dtype=torch.float32),
                        'labels': torch.tensor(target_labels, dtype=torch.int64)
                    })
                else:
                    targets_for_metric.append({
                        'boxes': torch.empty((0, 4), dtype=torch.float32),
                        'labels': torch.empty((0,), dtype=torch.int64)
                    })
            
            metric.update(preds_for_metric, targets_for_metric)

    print("Computing metrics (this may take a while)...")
    results = metric.compute()
    
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:15}: {v.item():.4f}")
        else:
            print(f"{k:15}: {v}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test SSD Model on COCO Dataset")
    parser.add_argument('--img_dir', type=str, required=True, help='Path to COCO validation images directory (e.g., val2017)')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to COCO validation annotations file (e.g., instances_val2017.json)')
    parser.add_argument('--weights', type=str, default=None, help='Path to trained model weights (.pth)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--conf_threshold', type=float, default=0.01, help='Confidence threshold for NMS')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='IoU threshold for NMS')
    
    args = parser.parse_args()
    main(args)
