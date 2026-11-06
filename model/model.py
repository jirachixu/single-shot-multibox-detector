import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from .anchor_boxes import *
from torchvision.ops import batched_nms

# debugging purposes
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SSDModel(nn.Module):
    '''
    SSD model using ResNet-50 as backbone. Architecture is described in the SSD paper.
        
    Args:
        **n_classes (int)**: Number of object classes (including background).
    '''
    def __init__(self, n_classes: int = 80) -> None:
        super().__init__()
        # +1 for background class
        self.n_classes = n_classes + 1
        self.n_anchors = 6 # number of anchors per feature map location
        # uses ResNet-50 to generate feature maps, we need to use the pretrained weights otherwise the 
        # feature maps will be meaningless
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.to(device)
        # layer1 contains non-feature-extracting layers
        self.layer1 = nn.Sequential(
            # remove avgpool and fully connected layers and last 3 conv blocks, since we need their outputs
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
        )
        # these layers extract the feature maps we will use for detection
        # dimensions specified by the input size of 300 x 300, as in the paper
        # https://cs231n.github.io/convolutional-networks/#conv:~:text=architectures%20section%20below.-,Convolution%20Demo,-.%20Below%20is%20a
        # demo of convolution operation on matrix
        self.layer2 = self.backbone.layer2 # 38 x 38
        self.layer3 = self.backbone.layer3 # 19 x 19
        self.layer4 = self.backbone.layer4 # 10 x 10
        self.extra1 = self._make_extra_layers(2048, 256, 512) # 5 x 5
        self.extra2 = self._make_extra_layers(512, 128, 256) # 3 x 3
        self.extra3 = self._make_extra_layers(256, 128, 256, padding=0, stride=1) # 1 x 1
        
        # detection heads that predict offsets for each feature map
        self.offset_convs = nn.ModuleList([
            # we multiply by 4 because we need to predict 4 offsets per anchor box
            # ∆cx, ∆cy, ∆w, ∆h
            # outputs will have shape (batch_size, n_anchors * 4, h, w)
            nn.Conv2d(512, self.n_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, self.n_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(2048, self.n_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, self.n_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, self.n_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, self.n_anchors * 4, kernel_size=3, padding=1),
        ])
        
        self.classification_convs = nn.ModuleList([
            # we multiply by the number of classes because we need to predict a class score for 
            # each anchor box per location
            # outputs will have shape (batch_size, n_anchors * n_classes, h, w)
            nn.Conv2d(512, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
            nn.Conv2d(2048, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
        ])
        
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Buffer.html
        # basically, a tensor that is not a parameter, but should still be part of the model state
        # no gradients, etc.
        self.register_buffer('anchor_boxes', None, persistent=False)
        
    def _initialize_anchors(self, feature_maps: list[torch.Tensor]) -> None:
        '''
        Initializes anchor boxes for each feature map size.
        
        Args:
            **feature_maps (list[torch.Tensor])**: List of feature map batches from different layers.
        '''
        if self.anchor_boxes is None:
            self.anchor_boxes = generate_anchor_boxes(feature_maps)
    
    def _make_extra_layers(self, 
            in_channels: int, 
            bottleneck_channels: int, 
            out_channels: int,
            padding: int = 1, 
            stride: int = 2
        ) -> nn.Sequential:
        '''
        Helper to create extra conv blocks, which are used to create the smallest feature maps. Based on the 
        bottleneck blocks in ResNet, which significantly reduce parameter count. The original SSD 
        architecture in the paper uses 2 conv layers instead, but using 3 here further decreases parameter
        count.
        
        Args:
            **in_channels (int)**: Number of input channels.
            **bottleneck_channels (int)**: Number of channels in the bottleneck layer.
            **out_channels (int)**: Number of output channels.
            **padding (int, optional)**: Padding for the 3x3 convolution.
            **stride (int, optional)**: Stride for the 3x3 convolution.
            
        Returns:
            **nn.Sequential**: Extra bottleneck convolutional block.
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass through the model. Turns the input image into feature maps, then applies the 
        corresponding detection heads for each feature map to get the final predictions.
        
        Args:
            **img (torch.Tensor)**: Input image tensor of shape (batch_size, 3, height, width).
            
        Returns:
            **tuple[torch.Tensor, torch.Tensor]**: Tuple containing offsets tensor and class scores tensor.
        '''
        feat = self.layer1(img)
        feat_1 = self.layer2(feat)
        feat_2 = self.layer3(feat_1)
        feat_3 = self.layer4(feat_2)
        feat_4 = self.extra1(feat_3)
        feat_5 = self.extra2(feat_4)
        feat_6 = self.extra3(feat_5)
        feature_maps = [feat_1, feat_2, feat_3, feat_4, feat_5, feat_6]
        
        self._initialize_anchors(feature_maps)
        
        all_offsets = []
        all_class_scores = []
        
        for i, feature_map in enumerate(feature_maps):
            offsets = self.offset_convs[i](feature_map)
            class_scores = self.classification_convs[i](feature_map)
            # we have to reshape here to allow for concatenation
            # torch.Tensor.contiguous() returns a contiguous tensor containing the same data 
            # as self tensor, which is more efficient than a non-contiguous tensor
            # also allows us to use the more efficient view() function instead of reshape()
            offsets = offsets.permute(0, 2, 3, 1).contiguous()
            class_scores = class_scores.permute(0, 2, 3, 1).contiguous()
            # (batch_size, n_anchors * h * w, 4)
            # flattens dimensions so that the middle dimension becomes each of the anchors' 
            # offsets at (0, 0), (0, 1), ..., (h, w), and then the 4 at the end takes every 
            # 4 consecutive values and groups them (∆cx, ∆cy, ∆w, ∆h)
            offsets = offsets.view(offsets.shape[0], -1, 4) 
            # (batch_size, n_anchors * h * w, n_classes)
            class_scores = class_scores.view(class_scores.shape[0], -1, self.n_classes)
            
            all_offsets.append(offsets)
            all_class_scores.append(class_scores)
        
        all_offsets = torch.cat(all_offsets, dim=1)
        all_class_scores = torch.cat(all_class_scores, dim=1)
        
        return all_offsets, all_class_scores
    
    def calculate_loss(self, 
            predicted_offsets: torch.Tensor, 
            predicted_class_scores: torch.Tensor, 
            true_boxes: torch.Tensor | list[torch.Tensor], 
            true_labels: torch.Tensor | list[torch.Tensor],
            neg_to_pos_ratio: int = 3
        ) -> torch.Tensor:
        '''
        Calculates the loss for the model. Combines localization loss (smooth L1) and classification loss 
        (cross-entropy). Exact formula described in the SSD paper.

        Args:
            **predicted_offsets (torch.Tensor)**: Predicted offsets tensor of shape (batch_size, n_anchors, 4).
            **predicted_class_scores (torch.Tensor)**: Predicted class scores tensor of shape (batch_size, 
            n_anchors, n_classes).
            **true_boxes (torch.Tensor)**: Ground truth bounding boxes tensor of shape (batch_size, n_boxes, 4).
            **true_labels (torch.Tensor)**: Ground truth labels tensor of shape (batch_size, n_boxes).
            **neg_to_pos_ratio (int, optional)**: Ratio of negative to positive samples for hard negative mining.
        
        Returns:
            **torch.Tensor**: Calculated loss value.
        '''
        batch_size = predicted_offsets.shape[0]
        tot_offset_loss = torch.tensor(0.0, device=device)
        tot_conf_loss = torch.tensor(0.0, device=device)
        n_pos_boxes = 0
        
        # we go image by image in the batch
        for batch in range(batch_size):
            true_boxes_batch = true_boxes[batch]
            assigned_boxes = assign_bounding_boxes(true_boxes_batch, self.anchor_boxes)
            mask = assigned_boxes >= 0
            
            n_pos_batch = mask.sum().item()
            n_pos_boxes += n_pos_batch
            
            encoded_boxes = encode_offsets(self.anchor_boxes, true_boxes_batch, assigned_boxes)
            
            # details of the math for loss functions can be found easily online
            if n_pos_batch > 0:
                # output is a 1x1 tensor (scalar), don't extract because we want to keep the gradient
                # if we don't use sum reduction and instead don't reduce, we would get an output 
                # tensor of shape (4 * n_pos_batch,), which is the loss for every single coordinate
                # of all assigned boxes
                # we also don't use mean because we regularize at the end
                tot_offset_loss += F.smooth_l1_loss(predicted_offsets[batch][mask], encoded_boxes[mask], reduction='sum')
            
            anchor_classes = torch.zeros(encoded_boxes.shape[0], device=device, dtype=torch.long)
            # gets the class labels for the assigned boxes, + 1 to account for background class at 0
            anchor_classes[mask] = true_labels[batch][assigned_boxes[mask]] + 1 
            # returns shape (n_anchors,) with the total cross entropy loss for each anchor box
            # the sum in the denominator of the softmax operation is over each of the 81 class 
            # scores in predicted_class_scores, the numerator is the class score corresponding
            # to the true class label in anchor_classes
            # reduction='none' returns the loss per element instead of averaging or summing
            conf_loss = F.cross_entropy(predicted_class_scores[batch], anchor_classes, reduction='none')
            # gets the cross_entropy loss for only the assigned (positive) anchor boxes and sums
            conf_loss_pos = conf_loss[mask].sum()
            # ~ operator is bitwise NOT, and mask is boolean so it will perform logical negation
            # and get the loss for only the unassigned (background) anchor boxes
            conf_loss_bg = conf_loss[~mask]
            # hard negative mining, keeping the highest loss background boxes
            # in other words, the background boxes that "look" most like actual objects
            # helps the model distinguish better between background and object
            if conf_loss_bg.shape[0] > 0 and n_pos_batch > 0:
                vals, idx = conf_loss_bg.sort(descending=True)
                n_negatives = min(neg_to_pos_ratio * n_pos_batch, conf_loss_bg.shape[0])
                hard_negatives = idx[:n_negatives]
                conf_loss_bg = conf_loss_bg[hard_negatives].sum()
            else:
                conf_loss_bg = torch.tensor(0.0, device=device)
            
            tot_conf_loss += conf_loss_pos + conf_loss_bg
            
        n_pos_boxes = max(n_pos_boxes, 1)
        return (tot_offset_loss + tot_conf_loss) / n_pos_boxes
    
    def nms(self, 
            boxes: torch.Tensor, 
            confidences: torch.Tensor, 
            iou_threshold: float = 0.5, 
            confidence_threshold: float = 0.4,
            max_detections_per_class: int = 100,
            use_torch: bool = True
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        '''
        Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes based on their scores. 
        This prevents overlapping boxes and removes duplicate boxes for the same object.
        
        Args:
            **boxes (torch.Tensor)**: Tensor of shape (n_boxes, 4) containing decoded bounding box coordinates.
            **confidences (torch.Tensor)**: Tensor of shape (n_boxes, n_classes) containing confidence scores 
            for each box.
            **iou_threshold (float, optional)**: IoU threshold for NMS.
            **confidence_threshold (float, optional)**: Confidence threshold to filter boxes before NMS.
            **max_detections_per_class (int, optional)**: Maximum number of detections to keep per class.
            **use_torch (bool, optional)**: Whether to use PyTorch's built-in NMS function. 
            If False, uses custom (slower) implementation. Default is True.
        
        Returns:
            **tuple(list)**: Tuple containing lists of kept boxes, labels, and scores after NMS. These 
            class labels are 1-indexed as the model represents the background class as 0, so 1 must be 
            subtracted to get the original class labels during prediction.
        '''
        if not use_torch:
            kept_boxes = []
            kept_labels = []
            kept_scores = []
            # do nms per class, except for class 0 which is background
            for class_n in range(1, self.n_classes):
                class_confidences = confidences[:, class_n]
                confidence_mask = class_confidences > confidence_threshold
                n_candidates = confidence_mask.sum().item()
                
                # don't waste time computing NMS if there are no boxes above the confidence threshold
                if n_candidates == 0:
                    continue
                
                if n_candidates > max_detections_per_class:
                    top_k_scores, top_k_indices = torch.topk(class_confidences, k=max_detections_per_class)
                    candidate_boxes = boxes[top_k_indices]
                    candidate_confidences = top_k_scores
                else:
                    candidate_boxes = boxes[confidence_mask]
                    candidate_confidences = class_confidences[confidence_mask]
                
                # stores indices of *candidate* boxes, not the original confidence tensor
                sorted_idx = torch.argsort(candidate_confidences, descending=True)
                
                keep_idx = []
                while sorted_idx.shape[0] > 0:
                    curr_max = sorted_idx[0]
                    keep_idx.append(curr_max.item())
                    # if there's only 1 item in sorted_idx, there's no other boxes to compare to
                    # so we break early
                    if sorted_idx.shape[0] == 1:
                        break
                    # we have to unsqueeze to match the expected input dimensions 
                    # for the iou function
                    curr_box = candidate_boxes[curr_max].unsqueeze(0)
                    remaining_boxes = candidate_boxes[sorted_idx[1:]]
                    # iou returns shape (1, n_remaining_boxes), which we squeeze to (n_remaining_boxes,)
                    ious = iou(curr_box, remaining_boxes).squeeze(0)
                    # get rid of boxes with iou above the threshold
                    below_threshold = ious <= iou_threshold
                    sorted_idx = sorted_idx[1:][below_threshold]
                
                if keep_idx:
                    keep_idx = torch.tensor(keep_idx, device=device)
                    kept_boxes.append(candidate_boxes[keep_idx])
                    kept_labels.append(torch.tensor([class_n] * len(keep_idx), device=device))
                    kept_scores.append(candidate_confidences[keep_idx])
            
            if kept_boxes:
                return (
                    torch.cat(kept_boxes, dim=0), 
                    torch.cat(kept_labels, dim=0), 
                    torch.cat(kept_scores, dim=0)
                )
            else:
                return None
        
        # gets the maximum confidence score and corresponding class for each box
        # max_scores, max_classes have shape (n_boxes,) and max_classes contains the 
        # class indices corresponding to the maximum scores from 0 to 79 since we started
        # from index 1 to ignore background
        max_scores, max_classes = confidences[:, 1:].max(dim=1)
        # filter out boxes whose maximum confidence score across all classes is lower
        # than the threshold
        confidence_mask = max_scores > confidence_threshold
        
        if confidence_mask.sum().item() == 0:
            return None
        
        candidate_boxes = boxes[confidence_mask]
        candidate_scores = max_scores[confidence_mask]
        # + 1 to account for background class at index 0
        candidate_classes = max_classes[confidence_mask] + 1 
        
        # indices of kept boxes
        keep = batched_nms(candidate_boxes, candidate_scores, candidate_classes, iou_threshold)
        
        return (
            candidate_boxes[keep],
            candidate_classes[keep],
            candidate_scores[keep]
        )

    def predict(self, 
            img: torch.Tensor, 
            iou_threshold: float = 0.5, 
            confidence_threshold: float = 0.4,
            max_detections_per_class: int = 100,
            use_torch: bool = True
        ) -> list[dict]:
        '''
        Makes predictions on the input image tensor(s) by performing a forward pass and applying NMS to 
        filter the results.
        
        Args:
            **img (torch.Tensor)**: Input image tensor of shape (batch_size, 3, height, width).
            **iou_threshold (float, optional)**: IoU threshold for NMS.
            **confidence_threshold (float, optional)**: Confidence threshold to filter boxes before NMS.
            **max_detections_per_class (int, optional)**: Maximum number of detections to keep per class.
            **use_torch (bool, optional)**: Whether to use PyTorch's built-in NMS function. 
            If False, uses custom (slower) implementation. Default is True.
            
        Returns:
            **list[dict]**: List of dictionaries containing detected boxes, labels, and scores for each 
            image in the batch.
        '''
        self.eval()
        # disables gradient calculation when evaluating/predicting for efficiency, since we only need 
        # them to train the model
        with torch.no_grad():
            predicted_offsets, predicted_class_scores = self.forward(img)
            batch_size = img.shape[0]
            all_detections = []
            for batch in range(batch_size):
                decoded_boxes = decode_offsets(self.anchor_boxes, predicted_offsets[batch])
                nms_result = self.nms(
                    decoded_boxes,
                    # we need to softmax to get actual class probabilities since that's what the 
                    # threshold value is with respect to
                    F.softmax(predicted_class_scores[batch], dim=1),
                    iou_threshold,
                    confidence_threshold,
                    max_detections_per_class,
                    use_torch
                )
                if nms_result is not None:
                    boxes, labels, scores = nms_result
                    detections = {
                        'boxes': boxes,
                        # subtract 1 to get original class labels since model uses 0 for background,
                        # which is not reflective of data (data doesn't have background class)
                        'labels': labels - 1,
                        'scores': scores
                    }
                    all_detections.append(detections)
                else:
                    all_detections.append({
                        'boxes': torch.empty((0, 4), device=device),
                        'labels': torch.empty((0,), device=device, dtype=torch.long),
                        'scores': torch.empty((0,), device=device)
                    })
        return all_detections

# test dimensionality of outputs
model = SSDModel()
model.to(device)

with torch.no_grad():
    x = torch.randn(2, 3, 300, 300, device=device)
    
    forwarded = model.forward(x)
    print(forwarded[0].shape, forwarded[1].shape)
    loss = model.calculate_loss(forwarded[0], 
                        forwarded[1], 
                        torch.randn(2, 5, 4, device=device), 
                        torch.randint(0, 80, (2, 5), device=device)
                    )
    print(f'loss: {loss}')
    decoded = decode_offsets(model.anchor_boxes, forwarded[0][0])
    print(decoded.shape)
    nms = model.nms(decoded, forwarded[1][0])
    print(f'nms: {nms}')
    if nms is not None:
        print(nms[0].shape, nms[1].shape, nms[2].shape)
        print(nms[1].min(), nms[1].max())
    
    x = model.layer1(x)
    
    x = model.layer2(x)
    print(f"Layer2 output: {x.shape}")
    x = model.layer3(x)
    print(f"Layer3 output: {x.shape}")
    x = model.layer4(x)
    print(f"Layer4 output: {x.shape}")
    x = model.extra1(x)
    print(f"Extra1 output: {x.shape}")
    x = model.extra2(x)
    print(f"Extra2 output: {x.shape}")
    x = model.extra3(x)
    print(f"Extra3 output: {x.shape}")
    
