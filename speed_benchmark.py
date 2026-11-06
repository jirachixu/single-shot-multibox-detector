import time
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou
from tqdm import tqdm

from model.model import SSDModel
from train_coco import COCODatasetWrapper, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def measure_speed(model, sample_tensor, runs=100, warmup=20):
    """Measures model inference speed (both raw forward pass and forward + NMS)."""
    print(f"\n--- Benchmarking Speed on {device.type.upper()} ---")
    model.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(sample_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.inference_mode():
        for _ in range(runs):
            _ = model(sample_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
                
    total_time = time.perf_counter() - start_time
    batch_size = sample_tensor.shape[0]
    avg_latency_ms = (total_time / (runs * batch_size)) * 1000
    fps = (runs * batch_size) / total_time
    print(f"Raw Model Forward Pass (Batch Size {batch_size}):")
    print(f"  Latency: {avg_latency_ms:.2f} ms per image")
    print(f"  Throughput: {fps:.1f} images/second (FPS)")

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.inference_mode():
        for _ in range(runs):
            _ = model.predict(sample_tensor, iou_threshold=0.45, confidence_threshold=0.3)
            if device.type == "cuda":
                torch.cuda.synchronize()
                
    total_time_e2e = time.perf_counter() - start_time
    avg_latency_e2e_ms = (total_time_e2e / (runs * batch_size)) * 1000
    fps_e2e = (runs * batch_size) / total_time_e2e
    print(f"End-to-End with NMS (Batch Size {batch_size}):")
    print(f"  Latency: {avg_latency_e2e_ms:.2f} ms per image")
    print(f"  Throughput: {fps_e2e:.1f} images/second (FPS)")


def evaluate_accuracy(model, dataset, subset_size=310, iou_thresh=0.5, conf_thresh=0.4):
    """
    Computes percentage of correctly identified objects:
    - Recall: % of ground truth objects that were found with correct class & IoU >= threshold.
    - Precision: % of predictions that matched a ground truth object.
    """
    print(f"\n--- Evaluating Detection Accuracy on {subset_size} images ---")
    eval_dataset = Subset(dataset, range(min(subset_size, len(dataset))))
    loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    total_gt_objects = 0
    total_predictions = 0
    correct_matches = 0

    model.eval()
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images_t = torch.stack(images).to(device)
            predictions = model.predict(images_t, iou_threshold=0.45, confidence_threshold=conf_thresh)

            for pred, target in zip(predictions, targets):
                gt_boxes = target['boxes'].to(device)      # [M, 4] normalized [0, 1]
                gt_labels = target['labels'].to(device)    # [M]
                num_gt = gt_boxes.shape[0]
                total_gt_objects += num_gt

                pred_boxes = pred['boxes']                 # [N, 4] normalized [0, 1]
                pred_labels = pred['labels']               # [N]
                num_pred = pred_boxes.shape[0]
                total_predictions += num_pred

                if num_gt == 0 or num_pred == 0:
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                matched_gt = set()

                # for each prediction, find if it matches an unmatched GT object
                for p_idx in range(num_pred):
                    best_iou, best_gt_idx = ious[p_idx].max(dim=0)
                    best_gt_idx = best_gt_idx.item()
                    best_iou = best_iou.item()

                    if (
                        best_iou >= iou_thresh
                        and best_gt_idx not in matched_gt
                        and pred_labels[p_idx] == gt_labels[best_gt_idx]
                    ):
                        matched_gt.add(best_gt_idx)
                        correct_matches += 1

    recall = (correct_matches / max(total_gt_objects, 1)) * 100
    precision = (correct_matches / max(total_predictions, 1)) * 100

    print("\n=== Accuracy Results (IoU >= 0.50, Conf >= 0.40) ===")
    print(f"Total Ground Truth Objects : {total_gt_objects}")
    print(f"Total Predictions Made     : {total_predictions}")
    print(f"Correct Detections (TP)    : {correct_matches}")
    print(f"Detection Recall           : {recall:.2f}%  (Found {correct_matches}/{total_gt_objects} objects)")
    print(f"Detection Precision        : {precision:.2f}%  ({correct_matches}/{total_predictions} detections were correct)")


if __name__ == '__main__':
    weights_path = "checkpoints/ssd_best.pth"

    dataset = COCODatasetWrapper(
        root="data/coco/val2017",
        annFile="data/coco/annotations/instances_val2017.json"
    )
    n_classes = len(dataset.coco_id_to_contiguous)

    model = SSDModel(n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    sample_single = torch.randn(1, 3, 300, 300, device=device)
    measure_speed(model, sample_single)

    sample_batch = torch.randn(8, 3, 300, 300, device=device)
    measure_speed(model, sample_batch)

    evaluate_accuracy(model, dataset, subset_size=310, iou_thresh=0.5, conf_thresh=0.4)