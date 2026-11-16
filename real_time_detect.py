import torch
from model.model import SSDModel
from test_coco import get_coco_category_map
from torchvision.datasets import CocoDetection
import cv2
import torchvision.transforms.functional as F_t

# Standard ImageNet normalization statistics required by pretrained ResNet50
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def draw_detections(frame, boxes, labels, scores, category_map):
    frame = cv2.resize(frame, (640, 480))
    frame_h, frame_w = frame.shape[:2]
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x1, y1, x2, y2 = box
            x1 = max(0, min(int(x1 * frame_w), frame_w - 1))
            y1 = max(0, min(int(y1 * frame_h), frame_h - 1))
            x2 = max(0, min(int(x2 * frame_w), frame_w - 1))
            y2 = max(0, min(int(y2 * frame_h), frame_h - 1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_name = category_map.get(int(label), 'Unknown')
            text_y = max(y1 - 10, 15)
            cv2.putText(frame, f"{label_name}: {score:.2f}", (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    coco_ds = CocoDetection(
        root='data/coco/val2017',
        annFile='data/coco/annotations/instances_val2017.json'
    )
    coco_id_to_contiguous, coco_contiguous_to_id = get_coco_category_map(coco_ds.coco)
    contiguous_to_name = {
        idx: coco_ds.coco.cats[coco_id]['name']
        for idx, coco_id in coco_contiguous_to_id.items()
    }
    
    model = SSDModel(n_classes=len(coco_id_to_contiguous))
    model.load_state_dict(torch.load('checkpoints/ssd_best.pth', map_location=device), strict=False)
    model.eval()
    model.to(device)
    print("Model loaded and set to evaluation mode.")

    webcam = cv2.VideoCapture(0)  # 0 refers to default webcam
    if not webcam.isOpened():
        raise Exception("Could not open webcam.")

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        _frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert from HWC to CHW format, add batch dimension, downscale, and normalize
        _frame = torch.from_numpy(_frame).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        _frame = torch.nn.functional.interpolate(
            _frame, size=(300, 300), mode='bilinear', align_corners=False, antialias=True
        )
        _frame = F_t.normalize(_frame, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        # Perform detection on the frame
        with torch.no_grad():
            detections = model.predict(_frame)
        
        for detection in detections:
            boxes = detection['boxes'].cpu().numpy()
            labels = detection['labels'].cpu().numpy()
            scores = detection['scores'].cpu().numpy()
            frame = draw_detections(frame, boxes, labels, scores, contiguous_to_name)

        cv2.imshow('Real-time Detection', frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.getWindowProperty('Real-time Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    webcam.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()