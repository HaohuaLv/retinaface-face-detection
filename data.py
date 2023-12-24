from datasets import load_dataset, DatasetDict
import numpy as np
import torch
import albumentations
from typing import Dict, List, Union, Tuple, Optional


def prepare_dataset(image_size: Tuple[int, int] = (840, 840)) -> DatasetDict:
    """

    Prepare widerface dataset from HuggingFace🤗 Dataset and preprocess it.

    Dataset Structure:

    {
    
        "pixel_values": Tensor(size=(3, height, width)),

        "labels": {
        
            "bboxes": Tensor(size=(num_bboxes, 4)),

            "num_bboxes": Tensor(size=(1,))

        }
    }

    Notice: bboxes is with YOLO format.

    """
    dataset = DatasetDict()
    dataset["train"] = load_dataset("wider_face", split="train")
    dataset["test"] = load_dataset("wider_face", split="validation")

    def remove_zero_bbox_filter_func(example):
        if len(example["bbox"]) == 0:
            return False
        for bbox in example["bbox"]:
            if 0 in bbox:
                return False
        return True

    dataset = dataset.filter(remove_zero_bbox_filter_func, input_columns="faces")

    train_transform = albumentations.Compose(
        [
            albumentations.RandomSizedBBoxSafeCrop(image_size[0], image_size[1]),
            albumentations.HorizontalFlip(),
            albumentations.Normalize(),
        ],
        bbox_params=albumentations.BboxParams(format="yolo", label_fields=["category"]),
    )

    val_transform = albumentations.Compose(
        [
            albumentations.Resize(image_size[0], image_size[1]),
            albumentations.Normalize(),
        ],
        bbox_params=albumentations.BboxParams(format="yolo", label_fields=["category"]),
    )

    def prepare_yolo_bbox_format(examples):
        width, height = examples["image"].size
        examples["bboxes"] = [
            [
                (bbox[0] + bbox[2] / 2) / width,
                (bbox[1] + bbox[3] / 2) / height,
                bbox[2] / width,
                bbox[3] / height,
            ]
            for bbox in examples["faces"]["bbox"]
        ]
        examples["category"] = [0] * len(examples["bboxes"])
        return examples

    dataset = dataset.map(
        prepare_yolo_bbox_format,
        desc="prepare_yolo_bbox_format",
        remove_columns="faces",
    )

    def transform_aug_ann(examples):
        pixel_values, labels = [], []
        for image, bboxes in zip(examples["image"], examples["bboxes"]):
            image = np.array(image.convert("RGB"))
            out = train_transform(
                image=image, bboxes=bboxes, category=[0] * len(bboxes)
            )

            pixel_values.append(torch.from_numpy(out["image"]).permute(2, 0, 1))
            labels.append(
                {
                    "bboxes": torch.tensor(out["bboxes"]),
                    "num_bboxes": torch.tensor(len(out["bboxes"])),
                }
            )

        return dict(pixel_values=pixel_values, labels=labels)

    def map_val(examples):
        pixel_values, labels = [], []
        for image, bboxes in zip(examples["image"], examples["bboxes"]):
            image = np.array(image.convert("RGB"))
            out = val_transform(image=image, bboxes=bboxes, category=[0] * len(bboxes))

            pixel_values.append(out["image"].transpose(2, 0, 1))
            labels.append({"bboxes": out["bboxes"], "num_bboxes": len(out["bboxes"])})

        return dict(pixel_values=pixel_values, labels=labels)

    dataset["train"] = dataset["train"].with_transform(transform_aug_ann)
    dataset["test"] = dataset["test"].map(
        map_val,
        batched=True,
        batch_size=16,
        remove_columns=dataset["train"].column_names,
        desc="val_transform",
    )

    dataset["test"] = dataset["test"].with_format("torch")

    return dataset
