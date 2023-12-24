from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput
import timm
from typing import Dict, List
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from math import ceil

from utils import box_iou, _center_to_corners, generalized_box_iou_loss


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(
            out_channel // 4, out_channel // 4, stride=1, leaky=leaky
        )
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.relu = nn.ReLU()

    def forward(self, input: Tensor) -> Tensor:
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = self.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1

        self.output_conv_list = nn.ModuleList(
            [
                conv_bn1X1(in_channels, out_channels, stride=1, leaky=leaky)
                for in_channels in in_channels_list
            ]
        )
        self.merge_conv_list = nn.ModuleList(
            [
                conv_bn(out_channels, out_channels, leaky=leaky)
                for i in range(len(in_channels_list) - 1)
            ]
        )

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        outputs = [
            output_conv(feature)
            for output_conv, feature in zip(self.output_conv_list, features)
        ]

        merges = [
            F.interpolate(up, size=down.shape[-2:]) + down
            for down, up in zip(outputs[:-1], outputs[1:])
        ]

        merges = [
            merge_conv(merge) for merge_conv, merge in zip(self.merge_conv_list, merges)
        ]

        outputs = merges + [outputs[-1]]

        return outputs


class YoloModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        assert (
            len(set([len(anchor_ratio) for anchor_ratio in config.anchors_ratios])) <= 1
        )
        assert len(config.anchors_ratios) == len(
            config.feature_extrator_kwargs["out_indices"]
        )
        assert len(config.image_size) == 2

        super().__init__(config)
        self.config = config
        self.feature_extractor = timm.create_model(**config.feature_extrator_kwargs)
        fpn_in_channels_list = [
            self.feature_extractor.feature_info.info[i]["num_chs"]
            for i in self.feature_extractor.feature_info.out_indices
        ]
        fpn_out_channels = config.fpn_out_channels

        self.fpn = FPN(fpn_in_channels_list, fpn_out_channels)

        self.sshs = nn.ModuleList(
            [SSH(fpn_out_channels, fpn_out_channels) for _ in fpn_in_channels_list]
        )

        self.class_heads = nn.ModuleList(
            [
                ClassHead(fpn_out_channels, len(config.anchors_ratios[0]))
                for _ in fpn_in_channels_list
            ]
        )

        self.bbox_heads = nn.ModuleList(
            [
                BboxHead(fpn_out_channels, len(config.anchors_ratios[0]))
                for _ in fpn_in_channels_list
            ]
        )

        self._prepare_anchors()

    def _prepare_anchors(self):
        def _get_anchors(nums_grids, anchors_ratios):
            def _get_anchor(num_grids, anchor_ratio):
                grid_size = [1 / ng for ng in num_grids]
                x_center = torch.arange(0.5 * grid_size[0], 1, 1 * grid_size[0])
                y_center = torch.arange(0.5 * grid_size[1], 1, 1 * grid_size[1])
                x_center, y_center = torch.meshgrid(x_center, y_center, indexing="xy")

                anchor_size = [anchor_ratio * gs for gs in grid_size]
                width = torch.full_like(x_center, anchor_size[0])
                height = torch.full_like(y_center, anchor_size[1])

                zeros = torch.zeros_like(x_center)
                x_half_gs = torch.full_like(x_center, 0.5 * grid_size[0])
                y_half_gs = torch.full_like(y_center, 0.5 * grid_size[1])

                anchors = torch.stack((x_center, y_center, width, height), dim=-1)
                anchors_weight = torch.stack(
                    (x_half_gs, y_half_gs, width, height), dim=-1
                )
                anchors_bias = torch.stack((x_center, y_center, zeros, zeros), dim=-1)

                return anchors, anchors_weight, anchors_bias

            anchors, anchors_weight, anchors_bias = [], [], []
            for num_grids, anchor_ratios in zip(nums_grids, anchors_ratios):
                anchors_grid, weight_grid, bias_grid = [], [], []
                for anchor_ratio in anchor_ratios:
                    anchor, weight, bias = _get_anchor(num_grids, anchor_ratio)
                    anchors_grid.append(anchor)
                    weight_grid.append(weight)
                    bias_grid.append(bias)
                anchors.append(torch.concatenate(anchors_grid, dim=-1).view(-1, 4))
                anchors_weight.append(
                    torch.concatenate(weight_grid, dim=-1).view(-1, 4)
                )
                anchors_bias.append(torch.concatenate(bias_grid, dim=-1).view(-1, 4))
            anchors = torch.concatenate(anchors, dim=0)
            anchors_weight = torch.concatenate(anchors_weight, dim=0)
            anchors_bias = torch.concatenate(anchors_bias, dim=0)

            return anchors, anchors_weight, anchors_bias

        self.reductions = [
            self.feature_extractor.feature_info.info[i]["reduction"]
            for i in self.feature_extractor.feature_info.out_indices
        ]

        nums_grids = [
            (
                ceil(self.config.image_size[0] / reduction),
                ceil(self.config.image_size[1] / reduction),
            )
            for reduction in self.reductions
        ]

        anchors, anchors_weight, anchors_bias = _get_anchors(
            nums_grids, self.config.anchors_ratios
        )
        self.register_buffer("anchors", anchors)
        self.register_buffer("anchors_weight", anchors_weight)
        self.register_buffer("anchors_bias", anchors_bias)

    def _post_process_bboxes(self, pred_bboxes: Tensor) -> Tensor:
        return pred_bboxes * self.anchors_weight.unsqueeze(
            0
        ) + self.anchors_bias.unsqueeze(0)

    def _pre_process_labels(self, labels):
        bboxes_target, logits_target = [], []
        for label in labels:
            ref_bboxes = label["bboxes"]
            iou = box_iou(
                _center_to_corners(self.anchors), _center_to_corners(ref_bboxes)
            )[0]

            bbox_target = ref_bboxes[iou.argmax(dim=1)]

            maxiou_per_anchor = iou.max(dim=0)
            difficult_filter = maxiou_per_anchor.indices[maxiou_per_anchor.values > 0.2]
            mask = iou.max(dim=1).values >= 0.35
            mask[difficult_filter] = True
            logit_target = mask.type(torch.long)

            bboxes_target.append(bbox_target)
            logits_target.append(logit_target)

        bboxes_target = torch.stack(bboxes_target, dim=0)
        logits_target = torch.stack(logits_target, dim=0)

        return bboxes_target, logits_target

    def stack_flatten_to_imgsize(self, flatten: Tensor) -> Tensor:
        start_reduction_ids = 0
        stacken = []
        for reduction in self.reductions:
            reduction_iou_shape = [
                ceil(shape / reduction) for shape in self.config.image_size
            ]
            num_anchors = (reduction_iou_shape[0] * reduction_iou_shape[1]) * len(
                self.config.anchors_ratios[0]
            )
            reduction_ious = flatten[
                start_reduction_ids : start_reduction_ids + num_anchors
            ]
            reduction_iou = (
                reduction_ious.view(
                    1, 1, *reduction_iou_shape, len(self.config.anchors_ratios[0])
                )
                .max(dim=-1)
                .values
            )
            stacken.append(
                F.interpolate(
                    reduction_iou, self.config.image_size, mode="nearest"
                ).view(self.config.image_size)
            )
            start_reduction_ids += num_anchors

        assert start_reduction_ids == flatten.shape[-1]
        stacken = torch.stack(stacken, dim=-1)
        stacken = torch.max(stacken, dim=-1).values
        return stacken

    def forward(
        self, pixel_values: Tensor, labels: List[Dict[str, torch.Tensor]] = None
    ) -> ModelOutput:

        features = self.feature_extractor(pixel_values)

        fpn_outputs = self.fpn(features)

        ssh_outputs = [
            ssh(fpn_output) for ssh, fpn_output in zip(self.sshs, fpn_outputs)
        ]

        pred_logits = torch.cat(
            [
                class_head(ssh_output)
                for class_head, ssh_output in zip(self.class_heads, ssh_outputs)
            ],
            dim=1,
        )

        pred_bboxes = torch.cat(
            [
                bbox_head(ssh_output)
                for bbox_head, ssh_output in zip(self.bbox_heads, ssh_outputs)
            ],
            dim=1,
        )
        # pred_bboxes[..., :2] = pred_bboxes[..., :2].tanh()
        pred_bboxes[..., 2:4] = pred_bboxes[..., 2:4].exp()
        pred_bboxes = self._post_process_bboxes(pred_bboxes)

        if labels is not None:
            self.logit_criterion = nn.CrossEntropyLoss()
            self.bbox_criterion = generalized_box_iou_loss
            # self.bbox_criterion = nn.SmoothL1Loss()

            bboxes_target, logits_target = self._pre_process_labels(labels)

            logit_loss = self.logit_criterion(
                pred_logits.view(-1, 2), logits_target.view(-1).detach()
            )

            # bbox_loss = torch.zeros_like(logit_loss)
            # for i in range(pred_bboxes.shape[0]):
            #     if (logits_target[i] == 0).all():
            #         continue
            #     bbox_loss += self.bbox_criterion(
            #         _center_to_corners(pred_bboxes[i][logits_target[i] == 1]),
            #         _center_to_corners(bboxes_target[i][logits_target[i] == 1].detach()),
            #     ) / pred_bboxes.shape[0]

            if (logits_target == 0).all():
                bbox_loss = torch.zeros_like(logit_loss)
            else:
                bbox_loss = self.bbox_criterion(
                    _center_to_corners(pred_bboxes[logits_target == 1]),
                    _center_to_corners(bboxes_target[logits_target == 1].detach()),
                )

            loss = sum(
                [
                    loss_p * weight
                    for loss_p, weight in zip(
                        [logit_loss, bbox_loss],
                        self.config.lb_loss_weights,
                    )
                ]
            )

            return ModelOutput(
                loss=loss,
                pred_logits=pred_logits,
                pred_bboxes=pred_bboxes,
            )

        return ModelOutput(pred_logits=pred_logits, pred_bboxes=pred_bboxes)
