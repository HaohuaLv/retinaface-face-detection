import onnxruntime as ort
import numpy as np
from numpy import ndarray
import cv2
from typing import Tuple


def box_area(boxes: ndarray) -> ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def np_box_iou(boxes1: ndarray, boxes2: ndarray) -> ndarray:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = np.clip(right_bottom - left_top, a_min=0, a_max=np.inf)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def np_nms(
    bboxes: ndarray, confidences: ndarray, nms_threshold: float = 0.4
) -> ndarray:
    sort_indices = np.argsort(confidences)
    sorted_confidences, sorted_bboxes = confidences[sort_indices], bboxes[sort_indices]

    i = 0
    while i < sorted_bboxes.shape[0]:
        iou = np_box_iou(
            sorted_bboxes[[i], :],
            sorted_bboxes[i + 1 :],
        ).flatten()
        drop_bboxes_indices = np.where(iou >= nms_threshold)[0] + i + 1
        sorted_bboxes = np.delete(sorted_bboxes, drop_bboxes_indices, axis=0)
        sorted_confidences = np.delete(sorted_confidences, drop_bboxes_indices, axis=0)
        i += 1
    return sorted_bboxes, sorted_confidences


def _np_center_to_corners(bboxes_center: ndarray) -> ndarray:
    center_x, center_y, width, height = np.split(bboxes_center, 4, axis=-1)
    bbox_corners = np.concatenate(
        # top left x, top left y, bottom right x, bottom right y
        [
            (center_x - 0.5 * width),
            (center_y - 0.5 * height),
            (center_x + 0.5 * width),
            (center_y + 0.5 * height),
        ],
        axis=-1,
    )
    return bbox_corners


def np_softmax(np_inputs: ndarray) -> ndarray:
    exp = np.exp(np_inputs)
    exp_sum = np.sum(exp, axis=-1, keepdims=True)
    return exp / exp_sum


def post_process(
    pred_logits: ndarray,
    pred_bboxes: ndarray,
    confidence_threshold: float = 0.6,
    nms_threshold: float = 0.4,
    target_size: Tuple[int, int] = None,
) -> Tuple[ndarray, ndarray]:
    pred_confidences = np_softmax(pred_logits)[..., -1]
    pred_masks = pred_confidences > confidence_threshold

    pred_bboxes = _np_center_to_corners(pred_bboxes)

    result_confidences, result_bboxes = (
        pred_confidences[pred_masks],
        pred_bboxes[pred_masks],
    )

    result_bboxes, result_confidences = np_nms(
        result_bboxes, result_confidences, nms_threshold
    )

    if target_size is not None:
        result_bboxes = (result_bboxes * np.array(target_size * 2)).astype(np.int64)
    
    return result_confidences, result_bboxes


ort_session = ort.InferenceSession(
    "export/retina-backbone_mobilenetv2_050-ft_widerface-2.onnx",
    providers=["CUDAExecutionProvider"],
)

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Camera is not opened !"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to load frame from camera.")
        break
    
    pixel_values = frame.transpose(2, 0, 1).astype(np.float32) / 255
    pred_logits, pred_bboxes = ort_session.run(None, {"pixel_values": pixel_values})

    image_height, image_width = frame.shape[:2]
    result_confidences, result_bboxes = post_process(pred_logits, pred_bboxes, target_size=(image_width, image_height))

    for confidence, bbox in zip(result_confidences.tolist(), result_bboxes.tolist()):
        cv2.rectangle(frame, bbox[:2], bbox[2:], color=(0, 0, 255), thickness=2)
        cv2.putText(frame, f"{confidence:.3f}", bbox[:2], fontFace=1, fontScale=3, color=(0, 0, 255), thickness=2)
        cv2.rectangle(frame, (bbox[2], (bbox[3] - int(confidence*(bbox[3] - bbox[1])))), (bbox[2]+20, bbox[3]+1), color=(0, 0, 255), thickness=-1)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
