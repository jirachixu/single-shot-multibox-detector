import torch
from PIL import ImageDraw
from train_coco import COCODatasetWrapper
from model.model import SSDModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = COCODatasetWrapper(
    root="data/coco/val2017",
    annFile="data/coco/annotations/instances_val2017.json"
)
_, contiguous_to_coco = (
    dataset.coco_id_to_contiguous,
    {i: cid for cid, i in dataset.coco_id_to_contiguous.items()}
)

# get raw PIL image for drawing, and normalized tensor for prediction
real_idx = dataset.valid_indices[0]
raw_img, _ = dataset.coco_ds[real_idx]
orig_w, orig_h = raw_img.size

img_tensor, gt_target = dataset[0]
img_batch = img_tensor.unsqueeze(0).to(device)

model = SSDModel(n_classes=len(dataset.coco_id_to_contiguous)).to(device)
model.load_state_dict(torch.load("checkpoints/ssd_best.pth", map_location=device), strict=False)

detections = model.predict(img_batch, iou_threshold=0.45, confidence_threshold=0.5)[0]

draw = ImageDraw.Draw(raw_img)
boxes = detections["boxes"].cpu()
labels = detections["labels"].cpu()
scores = detections["scores"].cpu()

print(f"Detected {len(boxes)} objects above 0.5 confidence:")
for box, label, score in zip(boxes, labels, scores):
    xmin = box[0].item() * orig_w
    ymin = box[1].item() * orig_h
    xmax = box[2].item() * orig_w
    ymax = box[3].item() * orig_h
    
    coco_cat_id = contiguous_to_coco[label.item()]
    cat_name = dataset.coco_ds.coco.cats[coco_cat_id]["name"]
    print(f"  - {cat_name}: {score.item():.2%}")
    
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    draw.text((xmin + 4, ymin + 4), f"{cat_name} {score.item():.2f}", fill="red")

raw_img.save("sanity_check_prediction.jpg")
print("Saved visual prediction to sanity_check_prediction.jpg!")