import torch
from torch import Tensor, nn
from torchvision import transforms
import torchvision.transforms.functional as F

from model import YoloModel


class E2E_YoloModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = YoloModel.from_pretrained(
            # "checkpoints/retina-backbone_mobilenetv2_050-ft_widerface/checkpoint-19000"
            # "checkpoints/retina-backbone_resnet50-ft_widerface/checkpoint-75900"
            "checkpoints/retina-backbone_mobilenetv2_050-ft_widerface-2/checkpoint-3610"
        )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.model.config.image_size),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def forward(self, pixel_values: Tensor):
        # assert len(pixel_values.shape) == 3 and pixel_values.shape[0] == 3
        print(pixel_values.shape)
        print(self.model.config.image_size)
        pixel_values = self.image_transform(pixel_values.unsqueeze(0))
        print(pixel_values.shape)
        return self.model(pixel_values)


model = E2E_YoloModel()
image_height, image_width = 1000, 1000
torch_input = torch.randn(3, image_height, image_width, requires_grad=True)
# print(model(torch_input)["pred_logits"].shape)
# image_height, image_width = 480, 640
# torch_input = torch.randn(3, image_height, image_width, requires_grad=True)
# print(model(torch_input)["pred_logits"].shape)
torch.onnx.export(
    model,
    torch_input,
    # "export/retina-backbone_mobilenetv2_050-ft_widerface.onnx",
    # "export/retina-backbone_resnet50-ft_widerface.onnx",
    "export/retina-backbone_mobilenetv2_050-ft_widerface-2.onnx",
    export_params=True, 
    do_constant_folding=True,
    input_names=["pixel_values"],
    output_names=["pred_logits", "pred_bboxes"],
    dynamic_axes={"pixel_values": {1: "image_height", 2: "image_width"}},
)

import onnxruntime as ort

ort_session = ort.InferenceSession(
    "export/retina-backbone_mobilenetv2_050-ft_widerface.onnx",
    provider_options=["CPUExecutionProvider"],
)
# onnx_input = torch_input.numpy()
import numpy as np
image_height, image_width = 480, 640
onnx_input = np.random.randn(3, image_height, image_width).astype(dtype=np.float32)

onnxruntime_outputs = ort_session.run(None, {"pixel_values": onnx_input})

pass
