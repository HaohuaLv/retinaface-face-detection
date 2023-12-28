import timm
from model import YoloModel
from transformers import PretrainedConfig
import onnxruntime
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from argparse import ArgumentParser
from torch import Tensor
import os
from tqdm import tqdm


from data import prepare_dataset
from utils import box_iou, _center_to_corners


def nms2(
    bboxes: Tensor, confidences: Tensor, nms_threshold: float = 0.4, topk=5000
) -> Tensor:
    if topk > confidences.shape[0]:
        topk = confidences.shape[0]
    sorted_confidences, sort_indices = confidences.topk(k=topk)
    sorted_bboxes = bboxes[sort_indices]
    i = 0
    while i < sorted_bboxes.shape[0]:
        iou = box_iou(
            _center_to_corners(sorted_bboxes[i].unsqueeze(0)),
            _center_to_corners(sorted_bboxes[i + 1 :]),
        )[0].flatten()
        drop_bboxes_indices = torch.where(iou >= nms_threshold)[0] + i + 1
        sorted_bboxes = np.delete(sorted_bboxes, drop_bboxes_indices, axis=0)
        sorted_confidences = np.delete(sorted_confidences, drop_bboxes_indices, axis=0)
        i += 1
    return sorted_bboxes, sorted_confidences


def inference(iou_threshold: float = 0.5):
    items = []

    image_id, gt_id = 0, 0
    tqdm_dataloader = tqdm(dataloader)
    for batch in tqdm_dataloader:
        pixel_values, labels = batch["pixel_values"], batch["labels"]
        with torch.no_grad():
            outputs = model(pixel_values.cuda())
        pred_logits = outputs["pred_logits"].cpu().softmax(dim=-1)[..., -1]
        pred_bboxes = outputs["pred_bboxes"].cpu()

        for i in range(pred_logits.shape[0]):
            batch_pred_bboxes = pred_bboxes[i].view(-1, 4)
            batch_pred_logits = pred_logits[i].view(-1)
            ref_bboxes = labels[i]["bboxes"]

            batch_result_bboxes, batch_result_confidences = nms2(
                batch_pred_bboxes, batch_pred_logits
            )

            # iou: Tensor(shape=[num_result_bboxes, num_ref_bboxes])
            iou = box_iou(
                _center_to_corners(batch_result_bboxes), _center_to_corners(ref_bboxes)
            )[0]
            iou_max_per_result, iou_max_indices = iou.max(dim=1)
            iou_max_indices[iou_max_per_result < iou_threshold] = -1

            batch_result_confidences = batch_result_confidences.tolist()
            iou_max_per_result, iou_max_indices = (
                iou_max_per_result.tolist(),
                iou_max_indices.tolist(),
            )
            items.extend(
                [
                    [image_id, ref_gt_id + gt_id, confidence, iou]
                    for ref_gt_id, confidence, iou in zip(
                        iou_max_indices,
                        batch_result_confidences,
                        iou_max_per_result,
                    )
                ]
            )
            gt_id += ref_bboxes.shape[0]
            image_id += 1

    items = torch.tensor(items)
    items = items[items[:, 2].sort(descending=True).indices]
    print(items.shape)

    num_gt = len(set(items[:, 1].tolist()))
    items = items.tolist()
    precisions, recalls = [], []
    TP = 0
    recall_gt_set = set()
    for i, item in enumerate(items):
        image_id, gt_id, confidence, iou = item

        if gt_id != -1:
            TP += 1
            recall_gt_set.add(gt_id)

        # precision = TP / (i+1)
        precision = len(recall_gt_set) / (i+1)
        recall = len(recall_gt_set) / num_gt
        precisions.append(precision)
        recalls.append(recall)
    
    max_precisions = 0
    precisions.reverse()
    for i, precision in enumerate(precisions):
        if precision > max_precisions:
            max_precisions = precision
        precisions[i] = max_precisions
    precisions.reverse()

    AP = np.trapz(precisions, recalls)
    print(f"AP={AP}, ")
    import matplotlib.pyplot as plt
    plt.plot(recalls, precisions)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.show()


parser = ArgumentParser(description="Retinaface Inference")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    # default="checkpoints/retina-backbone_resnet50-ft_widerface/checkpoint-51612",
    default="checkpoints/retina-backbone_mobilenetv2_050-ft_widerface/checkpoint-19000",
)
args = parser.parse_args()


def collect_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    return dict(pixel_values=pixel_values, labels=labels)


model = YoloModel.from_pretrained(
    args.checkpoint_path,
    config=os.path.join(args.checkpoint_path, "config.json"),
    local_files_only=True,
)
dataset = prepare_dataset(model.config.image_size)
# dataset["test"] = dataset["test"].shard(10, 0)
print(len(dataset["test"]))
dataloader = DataLoader(dataset["test"], collate_fn=collect_fn, batch_size=64)
model.eval().cuda()
inference()
