import matplotlib.pyplot as plt
from PIL import ImageDraw
from model import YoloModel
import torch
from PIL import ImageDraw
import torchvision.transforms.functional as F
from torch import Tensor
from argparse import ArgumentParser
import os

from data import prepare_dataset
from utils import box_iou, _center_to_corners, nms


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
model.eval()


mean = torch.tensor((0.485, 0.456, 0.406)).view(-1, 1, 1)
std = torch.tensor((0.229, 0.224, 0.225)).view(-1, 1, 1)


def denorm_image(pixel_values: Tensor) -> Tensor:
    return pixel_values * std + mean


for data in dataset["test"]:
    img = F.to_pil_image(denorm_image(data["pixel_values"]))

    with torch.no_grad():
        outputs = model(data["pixel_values"].unsqueeze(0))

    logits_stacken = model.stack_flatten_to_imgsize(
        outputs.pred_logits.softmax(dim=-1)[..., -1].view(-1)
    )

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(img)
    ax1.set_title("source image")

    ax2 = plt.subplot(2, 2, 2)
    pred_logits = outputs["pred_logits"].softmax(dim=-1)[..., -1]
    pred_masks = pred_logits > 0.5
    pred_bboxes = outputs["pred_bboxes"]
    result_bboxes = pred_bboxes[pred_masks]
    result_bboxes = nms(result_bboxes, pred_logits[pred_masks])

    ref_bbox = data["labels"]["bboxes"]
    iou = box_iou(_center_to_corners(result_bboxes), _center_to_corners(ref_bbox))[0]
    precision = ((iou.max(dim=0).values >= 0.5).sum() / iou.shape[0]).item()
    recall = ((iou.max(dim=0).values >= 0.5).sum() / iou.shape[1]).item()
    print(f"precision={precision}, recall={recall}")

    result_bboxes = (result_bboxes * torch.tensor(img.size * 2)).type(torch.int64)
    result_bboxes = _center_to_corners(result_bboxes).tolist()
    img_draw = ImageDraw.Draw(img)
    for bbox in result_bboxes:
        img_draw.rectangle(bbox, width=3)
    ax2.imshow(img)
    ax2.set_title("image with predicted bboxes")

    ious_stacken = model.stack_flatten_to_imgsize(
        model._pre_process_labels([data["labels"]])[1].type(torch.float32).view(-1)
    )

    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(ious_stacken)
    ax3.set_title("stacken target logits")

    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(logits_stacken)
    ax4.set_title("stacken predicted logits")

    plt.show()
