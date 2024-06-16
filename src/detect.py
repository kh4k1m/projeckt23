import torch
import numpy as np
from torchvision.ops import batched_nms
import cv2
from PIL import Image
from typing import List

from t_utils import xywh2xyxy, validate_bbox


class Detector:
    def __init__(self, model, confidence=0.25, nms_thresh=0.35, num_classes=5, draw_bbox=False):
        self.model = model
        print(model.name)
        self.confidence = confidence
        self.nms_thresh = nms_thresh
        self.num_classes = num_classes
        self.draw_bbox = draw_bbox

    def __call__(self, img, size=None):
        self.model.conf = self.confidence
        if isinstance(img, torch.Tensor):
            prep_img = self.preprocessing(img.clone())
        else:
            prep_img = img
        result = self.model(prep_img, size=size) if size else self.model(prep_img)
        if isinstance(img, torch.Tensor):
            boxes, scores, labels, batch_index = self.postprocess(result)
            if self.draw_bbox:
                drawed_img = [self.draw_bounding_boxes(img[i], boxes=boxes, labels=[str(x.detach()) for x in labels], batch_index=batch_index, i=i) for i in range(len(img))]
                return drawed_img, boxes, scores, labels
            return img, boxes, scores, labels
        result.render()
        image = Image.fromarray(result.ims[0])
        boxes = result.xyxy
        scores = [x[:, 4] for x in result.pred]
        labels = [x[:, 5] for x in result.pred]
        return image, boxes, scores, labels
    
    def preprocessing(self, img):
        return img / 255.0
    
    def draw_bounding_boxes(self, img, boxes, labels, batch_index, i):
        boxes = boxes.cpu().numpy().astype(int)
        img = img.permute((1, 2, 0)).numpy().astype(np.uint8)
        h, w, = img.shape[:2]
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = validate_bbox(x1, y1, x2, y2, w, h)
            if batch_index[idx] == i:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        return torch.from_numpy(img)
    
    def postprocess(self, det, confidence=0.25):
        
        label_offset = 5
        num_anchors = det.shape[1]
        box_preds = xywh2xyxy(det[:, :, :4])
        conf = det[:, :, 4]
        cls_preds = det * conf[:, :, None]
        num_classes = cls_preds.shape[-1] - label_offset
        num_anchors = box_preds.shape[1]
        boxes = box_preds.unsqueeze(2).expand(-1, num_anchors, num_classes, 4).contiguous()
        scores = cls_preds[..., label_offset:].contiguous()
        boxes = boxes.view(-1, 4)
        scores = scores.view(-1)
        rows = torch.arange(len(box_preds), dtype=torch.long)[:, None]
        cols = torch.arange(num_classes, dtype=torch.long)[None, :]
        idxs = rows * num_classes + cols
        idxs = idxs.unsqueeze(1).expand(len(box_preds), num_anchors, num_classes)
        idxs = idxs.to(scores).view(-1)
        mask = scores >= confidence
        boxesf = boxes[mask].contiguous()
        scoresf = scores[mask].contiguous()
        idxsf = idxs[mask].contiguous()

        keep = batched_nms(boxesf, scoresf, idxsf, self.nms_thresh)

        boxes = boxesf[keep]
        scores = scoresf[keep]
        labels = idxsf[keep] % num_classes
        batch_index = idxsf[keep] // num_classes

        return boxes, scores, labels, batch_index