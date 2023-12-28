import numpy as np
import torch
from torch import Tensor


def generalized_box_iou_loss(boxes1: Tensor, boxes2: Tensor):
    """
    Negative giou.
    boxes1, boxes2 must have the same shape.
    """
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(
            f"boxes1 must be [x0, y0, x1, y1] (corner) format, but got {boxes1}"
        )
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(
            f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}"
        )
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    width_height = (right_bottom - left_top).clamp(min=0)
    inter = width_height[:, 0] * width_height[:, 1]

    union = area1 + area2 - inter
    iou = inter / union

    top_left = torch.min(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    width_height_c = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height_c[:, 0] * width_height_c[:, 1]

    return -torch.mean(iou - (area - union) / area)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(
            f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}"
        )
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(
            f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}"
        )
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def _center_to_corners(bboxes_center: Tensor) -> Tensor:
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [
            (center_x - 0.5 * width),
            (center_y - 0.5 * height),
            (center_x + 0.5 * width),
            (center_y + 0.5 * height),
        ],
        dim=-1,
    )
    return bbox_corners


def nms(bboxes: Tensor, confidences: Tensor, nms_threshold: float = 0.4, topk=5000) -> Tensor:
    if topk > confidences.shape[0]:
        topk = confidences.shape[0]
    sorted_bboxes = bboxes[confidences.topk(k=topk).indices]
    i = 0
    while i < sorted_bboxes.shape[0]:
        iou = box_iou(_center_to_corners(sorted_bboxes[i].unsqueeze(0)), _center_to_corners(sorted_bboxes[i+1:]))[0].flatten()
        drop_bboxes_indices = torch.where(iou >= nms_threshold)[0] + i + 1
        sorted_bboxes = np.delete(sorted_bboxes, drop_bboxes_indices, axis=0)
        i += 1
    return sorted_bboxes
    

    
