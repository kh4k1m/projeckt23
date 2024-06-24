import torch
import numpy as np
from typing import List


def prepare4streamlit(batch, height, width):
    new_width = int(height * 1.2)
    start_x = (new_width - width) // 2
    for img in batch:
        padded_img = torch.zeros((height, new_width, 3))
        padded_img[:, start_x:start_x+width, :] = img
        img = padded_img
    return batch


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def save_auto_annot(lbl_file, boxes, labels):
    result = ''
    for box, label in zip(boxes, labels):
        for b, l in zip(box, label):
            lbox = ' '.join(list(map(str, map(float, b))))
            result += f"{int(l)} {lbox}\n"
    with open(lbl_file, 'w') as output:
        output.write(result)

def validate_bbox(x1, y1, x2, y2, w, h):
    if x1 < 0: x1 = 0
    if x2 > w: x2 = w - 1
    if y1 < 0: y1 = 0
    if y2 > h: y2 = h - 1
    return x1, y1, x2, y2

def tensor2list(t):
    if isinstance(t, List):
        return t
    return t.tolist()
    