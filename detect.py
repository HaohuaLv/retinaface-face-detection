from PIL import ImageDraw
from model import YoloModel
import torch
from PIL import ImageDraw, Image
from torchvision import transforms
from argparse import ArgumentParser
import os

from utils import _center_to_corners, nms


parser = ArgumentParser(description="Retinaface Inference")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="checkpoints/retina-backbone_resnet50-ft_widerface/checkpoint-51612",
)
parser.add_argument(
    "--image_path",
    type=str,
    default="pic/detect_src.png",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="pic/detect_result.png",
)
args = parser.parse_args()

model = YoloModel.from_pretrained(
    args.checkpoint_path,
    config=os.path.join(args.checkpoint_path, "config.json"),
    local_files_only=True,
)
model.eval()

image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(model.config.image_size),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


image = Image.open(args.image_path).convert("RGB")
pixel_values = image_transform(image)
with torch.no_grad():
    outputs = model(pixel_values.unsqueeze(0))
pred_logits = outputs["pred_logits"].softmax(dim=-1)[..., -1]
pred_masks = pred_logits > 0.5
pred_bboxes = outputs["pred_bboxes"]
result_bboxes = pred_bboxes[pred_masks]
result_bboxes = nms(result_bboxes, pred_logits[pred_masks], threshold=0.3)
result_bboxes = (result_bboxes * torch.tensor(image.size * 2)).type(torch.int64)
result_bboxes = _center_to_corners(result_bboxes).tolist()

img_draw = ImageDraw.Draw(image)
for bbox in result_bboxes:
    img_draw.rectangle(bbox, width=3)

image.save(args.save_path)
