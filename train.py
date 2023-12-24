from model import YoloModel
import torch
from transformers import TrainingArguments, Trainer
from transformers import PretrainedConfig
from argparse import ArgumentParser
import os

from utils import box_iou, _center_to_corners, nms
from data import prepare_dataset

parser = ArgumentParser(description="Retinaface Training")
parser.add_argument(
    "--model_config_file", type=str, default="config/resnet_config.json"
)
parser.add_argument("--lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_train_epochs", type=int, default=100)
parser.add_argument("--save_dir", type=str, default="checkpoints")
args = parser.parse_args()


def collect_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    return dict(pixel_values=pixel_values, labels=labels)


def compute_metrics(eval_predict):
    num_bboxes = eval_predict.label_ids[0]["num_bboxes"].tolist()
    bboxes = eval_predict.label_ids[0]["bboxes"].tolist()
    bboxes.reverse()
    precisions, recalls = [], []
    assert sum(num_bboxes) == len(bboxes)
    pred_logits, pred_bboxes = eval_predict.predictions
    pred_logits, pred_bboxes = torch.from_numpy(pred_logits).softmax(dim=-1)[
        ..., -1
    ], torch.from_numpy(pred_bboxes)
    pred_masks = pred_logits > 0.5

    precisions, recalls = [], []
    for i, n in enumerate(num_bboxes):
        batch_pred_bboxes = pred_bboxes[i][pred_masks[i]]
        batch_pred_logits = pred_logits[i][pred_masks[i]]
        ref_bboxes = torch.tensor([bboxes.pop() for _ in range(n)])
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
    assert len(bboxes) == 0
    return dict(
        precision=sum(precisions) / len(precisions),
        recall=sum(recalls) / len(recalls),
    )


model = YoloModel._from_config(PretrainedConfig.from_json_file(args.model_config_file))
dataset = prepare_dataset(model.config.image_size)


training_args = TrainingArguments(
    os.path.join(
        args.save_dir,
        f"retina-backbone_{model.config.feature_extrator_kwargs['model_name']}-ft_widerface",
    ),
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=1,
    logging_steps=100,
    learning_rate=args.learning_rate,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    eval_accumulation_steps=100,
    report_to=["tensorboard"],
    fp16=True,
    remove_unused_columns=False,
)


trainer = Trainer(
    model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collect_fn,
    compute_metrics=compute_metrics,
)


trainer.train()
