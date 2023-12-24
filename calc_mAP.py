from model import YoloModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
import os

from data import prepare_dataset
from utils import box_iou, _center_to_corners, nms


def threshold_inference(threshold: float = 0.5):
    precisions, recalls = [], []
    tqdm_dataloader = tqdm(dataloader)
    tqdm_dataloader.set_description(f"Threshold={threshold}")
    for batch in tqdm_dataloader:
        pixel_values, labels = batch["pixel_values"], batch["labels"]
        with torch.no_grad():
            outputs = model(pixel_values.cuda())
        pred_logits = outputs["pred_logits"].cpu().softmax(dim=-1)[..., -1]
        pred_bboxes = outputs["pred_bboxes"].cpu()
        pred_masks = pred_logits > threshold

        for i in range(pred_logits.shape[0]):
            batch_pred_bboxes = pred_bboxes[i][pred_masks[i]]
            batch_pred_logits = pred_logits[i][pred_masks[i]]
            ref_bboxes = labels[i]["bboxes"]

            if batch_pred_bboxes.shape[0] == 0:
                precisions.append(0)
                recalls.append(0)
                continue

            batch_pred_bboxes = nms(batch_pred_bboxes, batch_pred_logits)
            iou = box_iou(
                _center_to_corners(batch_pred_bboxes), _center_to_corners(ref_bboxes)
            )[0]
            precision = ((iou.max(dim=0).values >= 0.5).sum() / iou.shape[0]).item()
            recall = ((iou.max(dim=0).values >= 0.5).sum() / iou.shape[1]).item()
            precisions.append(precision)
            recalls.append(recall)
        tqdm_dataloader.set_postfix(
            dict(
                precision=sum(precisions) / len(precisions),
                recall=sum(recalls) / len(recalls),
            )
        )

    threshold_precision = sum(precisions) / len(precisions)
    threshold_recall = sum(recalls) / len(recalls)

    return threshold_precision, threshold_recall


def collect_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    return dict(pixel_values=pixel_values, labels=labels)


parser = ArgumentParser(description="Retinaface Inference")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="checkpoints/retina-backbone_resnet50-ft_widerface/checkpoint-51612",
)
args = parser.parse_args()

model = YoloModel.from_pretrained(
    args.checkpoint_path,
    config=os.path.join(args.checkpoint_path, "config.json"),
    local_files_only=True,
)
dataset = prepare_dataset(model.config.image_size)
dataloader = DataLoader(dataset["test"], collate_fn=collect_fn, batch_size=64)
model.eval().cuda()


thresholds_map, precisions_map, recalls_map = [], [], []


threshold = 0.5
for i in range(2, 10):
    threshold_precision, threshold_recall = threshold_inference(threshold)

    thresholds_map.append(threshold)
    precisions_map.append(threshold_precision)
    recalls_map.append(threshold_recall)

    AP = threshold_precision * threshold_recall
    if AP <= 0.1:
        break
    threshold += 0.5**i

threshold = 0.25
for i in range(3, 10):
    threshold_precision, threshold_recall = threshold_inference(threshold)

    thresholds_map.append(threshold)
    precisions_map.append(threshold_precision)
    recalls_map.append(threshold_recall)

    AP = threshold_precision * threshold_recall
    if AP <= 0.1:
        break
    threshold -= 0.5**i


print(f"precisions_map={precisions_map}, recalls_map={recalls_map}")

import json

with open("mAP.json", "w") as f:
    json.dump(
        dict(
            thresholds_map=thresholds_map,
            precisions_map=precisions_map,
            recalls_map=recalls_map,
        ),
        f,
    )
