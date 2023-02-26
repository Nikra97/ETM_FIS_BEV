from typing import Tuple
from argparse import Namespace
# import datasets
# import datasets.samplers as samplers
# from datasets.data_prefetcher import data_prefetcher
# from datasets import build_dataset, get_coco_api_from_dataset
from collections import OrderedDict, defaultdict
import copy
from typing import Optional, List, Dict
import io
import math
from scipy.optimize import linear_sum_assignment

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch import Tensor
from torchvision.models._utils import IntermediateLayerGetter

from .ops.modules import MSDeformAttn
from mmcv.runner import force_fp32

import torch.utils.checkpoint as cp

from PIL import Image


from .deformable_detr_utils import (box_cxcywh_to_xyxy, generalized_box_iou, generalized_box_cdist,NestedTensor, nested_tensor_from_tensor_list,
                                                                                                  accuracy, get_world_size, interpolate,
                                                                                                  is_dist_avail_and_initialized, inverse_sigmoid, is_main_process) 


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, inputs, masks):  # feats, mask_list
        x = inputs  # tensor_list.tensors  # x = feats

        mask = masks  # tensor_list.mask # mask_list
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t,
                                     2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, inputs, masks):  # feats, mask_list
        x = inputs
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(position_embedding, hidden_dim):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(
                0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * \
                (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * \
                ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - \
                neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * \
                cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(
                c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# class ConvExtractor(nn.Module):
#     def __init__(self, inplanes, planes,stride=1 ,groups=1) -> None:
#         super().__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         self.conv2 = conv3x3(inplanes, planes, 2)
#         self.conv3 = conv3x3(inplanes, planes, 2)
#         self.conv4 = conv3x3(inplanes, planes, 2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#     def forward(self, x):
#         pass 
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_dnc=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out
        if x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, channel_list=[64, 128, 256, 512], dont_use_bev_input=False):
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        #dont_use_bev_input = True
        self.groups = groups
        self.base_width = width_per_group
        if dont_use_bev_input:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            
            self.conv1 = nn.Conv2d(64, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(64, self.inplanes, kernel_size=3, stride=2, padding=1,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(
            block, channel_list[0], layers[0])
        self.layer2 = self._make_layer(block, channel_list[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, channel_list[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, channel_list[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_list[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        #print(f"Input: {x.shape =}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(f"After Stem: {x.shape =}")
        x = self.layer1(x)
        #print(f"After Layer 1: {x.shape =}")
        x = self.layer2(x)
        #print(f"After Layer 2: {x.shape =}")
        x = self.layer3(x)
        #print(f"After Layer 3: {x.shape =}")
        x = self.layer4(x)
        #print(f"After Layer 4: {x.shape =}")

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        #model.load_state_dict(state_dict)
        #print("no pretrained model allowed :>")
        pass

    return model

# 2, 2, 2, 2


def resnet18(pretrained=False, progress=True, layers=[2, 2, 2, 2], **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, layers, pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, layers=[3, 4, 6, 3], **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool, num_feature_levels:int=4):
        super().__init__()
        if return_interm_layers:
            if num_feature_levels == 4:
                return_layers = {"layer1": "0", "layer2": "1",
                                "layer3": "2", "layer4": "3"}
                #return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [2, 4, 8, 16]  # [4, 8, 16, 32]
                # [256, 512, 1024, 2048]
                self.num_channels = [64, 128, 256, 512]  # [64, 128, 256, 512]
            elif num_feature_levels == 3:
                return_layers = {"layer2": "0",
                                "layer3": "1", "layer4": "2"}
                #return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [4, 8, 16]  # [4, 8, 16, 32]
                # [256, 512, 1024, 2048]
                self.num_channels = [128, 256, 512]  # [64, 128, 256, 512]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)

    def forward(self, inputs, masks):
        xs = self.body(inputs)
        out: Dict[str, Tuple] = {}  # torch.Tensor
        for name, x in xs.items():
            m = masks
            assert m is not None
            mask = F.interpolate(
                m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = (x, mask)  # Tuple
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 checkpoint: bool = False,
                 dcn: bool = False,
                 num_feature_levels: int = 4):
        norm_layer = FrozenBatchNorm2d
        if checkpoint or dcn:
            #print('Training with checkpoint to save GPU memory.')
            #from .resnet import resnet50, resnet101
            if dcn:
                print('Training with dcn.')
                stage_with_dcn = [False, True, True, True]
            else:
                stage_with_dcn = [False, False, False, False]
            backbone = eval(name)(
                replace_stride_with_dilation=[False, False, False],
                pretrained=is_main_process(), norm_layer=norm_layer, stage_with_dcn=stage_with_dcn)
        else:
            if name == "resnet18":
                backbone = resnet18(replace_stride_with_dilation=[
                                    False, False, False], norm_layer=norm_layer)
            elif name == "resnet34":
                backbone = resnet34(replace_stride_with_dilation=[
                                    False, False, False], norm_layer=norm_layer)
            elif name == "resnet50":
                backbone = resnet50(replace_stride_with_dilation=[
                                    False, False, False], norm_layer=norm_layer)
            elif name == "resnet101":
                backbone = resnet50(replace_stride_with_dilation=[
                                    False, False, dilation], norm_layer=norm_layer)
            else:
                raise ValueError("Backbone name not in config")

            # backbone = getattr(torchvision.models, name)(
            #     replace_stride_with_dilation=[False, False, dilation],
            #     pretrained=is_main_process(), norm_layer=norm_layer)

            #backbone = resnet_creation_func()
                #replace_stride_with_dilation=[False, False, dilation],
                #pretrained=False, norm_layer=norm_layer)
        super().__init__(backbone, train_backbone,
                         return_interm_layers, num_feature_levels)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, inputs, masks):
        xs = self[0](inputs, masks)
        out: List[Tuple] = []  # Tuple
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x[0], x[1]).to(x[0].dtype))

        return out, pos


def build_backbone(backbone='resnet18', layers=[
                   2, 2, 2, 2], return_feature_layers=True, position_embedding='sine', num_pos_feats=128,hidden_dim=1024, dilation=[
                                    False, False, False],num_feature_level=4):
    position_embedding = build_position_encoding(position_embedding, hidden_dim)
    train_backbone = True #args.lr_backbone > 0
    # return_interm_layers = args.masks or (args.num_feature_levels > 1)
    return_interm_layers = True #args.num_feature_levels > 1
    dilation= [False,False,False,False]
    backbone = Backbone(backbone, train_backbone,
                        return_interm_layers, dilation, num_feature_levels=num_feature_level)
    model = Joiner(backbone, position_embedding)
    return model#, backbone


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(
            encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        #print(f'Training with {activation}.')

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(
                _cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat(
                [valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (
            output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(
            ~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        #print( f"src_flatten {src_flatten.shape = } spatial_shapes  {spatial_shapes.shape = } level_start_index  {level_start_index.shape = } valid_ratios  {valid_ratios.shape = } lvl_pos_embed_flatten  {lvl_pos_embed_flatten.shape = } mask_flatten  {mask_flatten.shape = }")

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        #print(f"{memory.shape =}")
        seg_memory, seg_mask = memory[:, level_start_index[-1]
            :, :], mask_flatten[:, level_start_index[-1]:]
        #print(f"{seg_memory.shape =} {seg_mask.shape =}")
        seg_memory = seg_memory.permute(0, 2, 1).view(bs, c, h, w)
        seg_mask = seg_mask.view(bs, h, w)
        #print(f"after Permute/View {seg_memory.shape =} {seg_mask.shape =}")
        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](
                output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(
                self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        #print(f"tgt {tgt.shape = } reference_points  {reference_points.shape = }  memory  {memory.shape = }  spatial_shapes  {spatial_shapes.shape = } level_start_index  {level_start_index.shape = } valid_ratios  {valid_ratios.shape = } query_embed  {query_embed.shape = } mask_flatten  {mask_flatten.shape = }")

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, seg_memory, seg_mask
        return hs, init_reference_out, inter_references_out, None, None, seg_memory, seg_mask


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

   
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        tensor_with_pos = self.with_pos_embed(
            src, pos)
        src2 = self.self_attn(tensor_with_pos, reference_points,
                              src, spatial_shapes, level_start_index, padding_mask)
        #print(f"src2 shape {src2.shape = }")
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        #print(f"Transformer encoder output shape {src.shape = }")
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points,
                           spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        # print(
        #     f"Constructed Decoder Layer with: {d_model = },  {n_levels = },  {n_heads = } ,  {n_points = } ")
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        #print("decoder")
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(
            0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        #print(f"Self Attn: {tgt2.shape =}")
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        #print(f"Cross Attn: {tgt2.shape =}")
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        #print(f"Out Decoder: {tgt.shape =}")
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:,
                                                          :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src,
                           src_spatial_shapes, src_level_start_index, src_padding_mask)
            #print(f"Output Decoder {output.shape = }")
            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + \
                        inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[...,
                                                        :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output.unsqueeze(0), reference_points.unsqueeze(0)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def swish(x):
    return x * torch.sigmoid(x)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "silu":
        return swish
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(hidden_dim, nheads, enc_layers, dec_layers, 
                                 dim_feedforward, dropout_transformer, activation,
                                 num_feature_levels, dec_n_points, enc_n_points,
                                 num_queries, return_intermediate_dec=True):
    return DeformableTransformer(
        d_model=hidden_dim,
        nhead=nheads,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout_transformer,
        activation=activation,
        return_intermediate_dec=return_intermediate_dec,
        num_feature_levels=num_feature_levels,
        dec_n_points=dec_n_points,
        enc_n_points=enc_n_points,
        two_stage=False,
        two_stage_num_proposals=num_queries)


# try:
#     from panopticapi.utils import id2rgb, rgb2id
# except ImportError:
#     pass


small_resnet = False

def build_MaskHeadSmallConv(hidden_dim,nheads,fpns, output_convs):
    return MaskHeadSmallConv(hidden_dim+nheads, fpns, hidden_dim, output_convs)


class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(
            hidden_dim, hidden_dim, nheads, dropout=0)
        self.mask_head = MaskHeadSmallConv(
            hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def forward(self, inputs, masks):
        # if not isinstance(samples, NestedTensor):
        #     samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.detr.backbone(inputs, masks)

        bs = features[-1][0].shape[0]

        src, mask = features[-1]
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(
            src_proj, mask, self.detr.query_embed.weight, pos[-1])

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1],
               "pred_boxes": outputs_coord[-1]}
        if self.detr.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        seg_masks = self.mask_head(src_proj, bbox_mask, [
                                   features[2][0], features[1][0], features[0][0]])
        outputs_seg_masks = seg_masks.view(
            bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        out["pred_masks"] = outputs_seg_masks
        return out


class DeformableDETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False, small_resnet=True):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            #print('Training with freezing detection branch of deformable detr.')
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(
            hidden_dim, hidden_dim, nheads, dropout=0)

        if small_resnet:
            in_channels_list = [512, 256, 128, 64]
        else:
            in_channels_list = [1024, 512, 256]
        #print(f"{in_channels_list = }")
        self.mask_head = MaskHeadSmallConv(
            hidden_dim + nheads, in_channels_list, hidden_dim)  # [1024,512, 256, 128]
        input_proj_list = []

        for in_channels in in_channels_list:
            input_proj_list.append(
                nn.Conv2d(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        self.seg_input_proj = nn.ModuleList(input_proj_list)
        #print(self.seg_input_proj)
        #print(self.mask_head)

    def forward(self, inputs, masks):
        """ The forward expects a inputs and masks, which consists of:
               - inputs: batched images, of shape [batch_size x 3 x H x W]
               - masks: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # if not isinstance(samples, NestedTensor):
        #     samples = nested_tensor_from_tensor_list(samples)
        mask_input = masks
        features, pos = self.detr.backbone(inputs, masks)

        bs = features[-1][0].shape[0]  # batchsize

        srcs = []
        masks = []
        for l, feat in enumerate(features):

            src, mask = feat

            proj_src = self.detr.input_proj[l](src)

            srcs.append(proj_src)
            masks.append(mask)
            assert mask is not None
        if self.detr.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            
            for l in range(_len_srcs, self.detr.num_feature_levels):     
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1][0])
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = mask_input
                mask = F.interpolate(
                    m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](
                    src, mask).to(src.dtype)  # First layer
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.detr.two_stage:
            query_embeds = self.detr.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, seg_memory, seg_mask = self.detr.transformer(
            srcs, masks, pos, query_embeds)

        #print(f"hs {hs.shape = } init_reference  {init_reference.shape = }  enc_outputs_class  {enc_outputs_class = } enc_outputs_coord_unact  {enc_outputs_coord_unact = } seg_memory  {seg_memory.shape = } seg_mask  {seg_mask.shape = }")

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.detr.class_embed[lvl](hs[lvl])
            tmp = self.detr.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1]}
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(
                outputs_class, outputs_coord)

        if self.detr.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {
                'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(hs[-1], seg_memory, mask=seg_mask)

        input_projections = [(features[-1][0]),
                             (features[-2][0]), (features[-3][0]), features[-4][0]]  # [self.seg_input_proj[0](features[-1][0]),
        #                      self.seg_input_proj[1](features[-2][0]), self.seg_input_proj[2](features[-3][0])]

        # for ip in input_projections:
        #     print(f"{ip.shape = }")

        # feature pyramid stuff <-- this is where the shapes are relevant
        seg_masks = self.mask_head(srcs[-1], bbox_mask, input_projections)

        outputs_seg_masks = seg_masks.view(
            bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        out["pred_masks"] = outputs_seg_masks
        return out


# def build_output_convs(in_channels, output_dict={"instance_centerness": 1, "instance_offset": 2, "instance_flow": 2, "motion_segmentation": 1}):
#     for o in output_dict:
#         output_dict[o] = nn.Conv2d(in_channels, output_dict[o], kernel_size=1)
#     return output_dict

class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, output_dict=None):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4,
                      context_dim // 8, context_dim // 16, context_dim // 64, context_dim // 128]

        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])

        self.lay6 = torch.nn.Conv2d(inter_dims[4], inter_dims[4], 3, padding=1)
        self.gn6 = torch.nn.GroupNorm(8, inter_dims[4])

        self.out_lay = torch.nn.Conv2d(
            inter_dims[4], 1, 3, padding=1)  # <- This would be differen

        # if output_dict is not None:
        #     self.future_pred_layers = build_output_convs(
        #         inter_dims[4], output_dict)
        """ 
        outheads_
            - instance_centerness: 1x5x1x200x200 - BxFx1xHxW <- CUR F = N Future Steps 
            - instance_offset:     1x5x2x200x200 - BxFx2xHxW <- CUR F = N Future Steps 
            - instance_flow:       1x5x2x200x200 - BxFx2xHxW <- CUR F = N Future Steps 
            - motion_segmentation: 1x5x200x200   - BxFx1xHxW
            - semantic_map:        1x3x200x200   - Bx3xHxW  <- CUR T 
        """

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
        self.adapter4 = torch.nn.Conv2d(fpn_dims[3], inter_dims[4], 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):

        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

        x = torch.cat([expand(x, bbox_mask.shape[1]),
                      bbox_mask.flatten(0, 1)], 1)
        #print(f"First Expand: {x.shape = }")
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        #print(f"Before adapter1: {x.shape = }")
        cur_fpn = self.adapter1(fpns[0])
        #print(f"First cur_fpn: {cur_fpn.shape = }")
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
            #print(f"cur_fpn.size(0) != x.size(0): {cur_fpn.shape = }")
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        #print(f"Interpolutaion with expan: {x.shape = }")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        #print(f"Before adapter2: {x.shape = }")
        cur_fpn = self.adapter2(fpns[1])
        #print(f"2 adapter2: {cur_fpn.shape = }")
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
            #print(f"cur_fpn.size(0) != x.size(0): {cur_fpn.shape = }")

        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        #print(f"Interpolutaion with expan: {x.shape = }")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)
        #print(f"TBefore adapter3: {x.shape = }")

        cur_fpn = self.adapter3(fpns[2])
        #print(f"after adapter3: {cur_fpn.shape = }")
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
            #print(f"cur_fpn.size(0) != x.size(0): {cur_fpn.shape = }")
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        #print(f"Interpolutaion with expan: {x.shape = }")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        #print(f"Fourth Expand: {x.shape = }")
        cur_fpn = self.adapter4(fpns[3])
        #print(f"after adapter4: {cur_fpn.shape = }")
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
            #print(f"cur_fpn.size(0) != x.size(0): {cur_fpn.shape = }")
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        #print(f"Interpolutaion with expan: {x.shape = }")

       
        # print(f"Interpolutaion with expan: {x.shape = }")
        x = self.lay6(x)
        x = self.gn6(x)
        x = F.relu(x)
        #print(f"Fourth Expand: {x.shape = }")
        #x = F.interpolate(x, size=200, mode="nearest")

        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(
            k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads,
                    self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim //
                    self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw",
                               qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        #print(f"MH AttentionMap Shape {weights.shape = }")
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        #print(f"{max_h = }, {max_w = }")
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(orig_target_sizes)
        assert orig_target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = orig_target_sizes.unbind(1)

        #print(f"{img_h = }, {img_w = }")
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        #print(f" {scale_fct.shape = }")
        boxes = boxes * scale_fct[:, None, :]
        #print(f" {boxes.shape = }")
        out_mask = outputs["pred_masks"]
        #print(f" {out_mask.shape = }")
        B, R, H, W = out_mask.shape
        out_mask = out_mask.view(B, R, H * W)
        #print(f" {out_mask.shape = }")
        out_mask = torch.gather(
            out_mask, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, H * W))
        #print(f"After gather {out_mask.shape = }")
        outputs_masks = out_mask.view(B, 100, H, W).squeeze(2)

        outputs_masks = F.interpolate(outputs_masks, size=(
            max_h, max_w), mode="bilinear", align_corners=False)
        
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            interpol_tmp = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            )
            
            results[i]["masks"] = interpol_tmp.byte()

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(
                outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            
            # cur_masks = interpolate(cur_masks[None], to_tuple(
            #     size), mode="bilinear").squeeze(0)
            cur_masks = interpolate(cur_masks[:, None], to_tuple(
                size), mode="bilinear").squeeze(1)
            print(f"{cur_masks.shape=}")
            cur_boxes = box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            print(f"after flatten {cur_masks.shape=}")
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros(
                        (h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                # seg_img = Image.fromarray(
                #     id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(
                    size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(
                        seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                # m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(
                    1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append(
                    {"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(
                ), "segments_info": segments_info}
            preds.append(predictions)
        return preds


def build_seg_detr(detr, args, freeze_detr=False):
    if args.backbone != "resnet18" or "resnet34":
        small_resnet = True
    else:
        small_resnet = False
    output_convs = []  # TODO generate MAPS here for needed properties
    return DeformableDETRsegm(detr, freeze_detr=freeze_detr, small_resnet=True, output_convs=output_convs)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, tqa=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.past_query_embed = None
        self.temporal_queries_activated = tqa
        if self.temporal_queries_activated:
            self.temporal_query_projection = nn.Sequential(
                nn.Linear(num_queries, out_features=num_queries),
                nn.Dropout(p=0.1),
                nn.ReLU(),
            )

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            print("query_embed", self.query_embed)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        print("Input Projection \n", self.input_proj)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers +
                    1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(
                self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList(
                [self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, inputs, masks_input):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # if not isinstance(samples, NestedTensor):
        #     samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(inputs, masks_input)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1][0])  # .tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks_input
                mask = F.interpolate(
                    m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](src, mask).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
            if self.temporal_queries_activated:
                if self.past_query_embed is None:
                    self.past_query_embed = self.temporal_query_projection(
                        query_embeds)
                
                query_embeds += self.past_query_embed

        

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, seg_memory, seg_mask = self.transformer(
            srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {
                'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J]
                                     for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot,
                                     num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - \
                accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i]
                                 for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items(
        ) if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets,
                          indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



def build_detr(backbone , transformer,num_classes, args):
    return DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=False,#args.aux_loss,
        with_box_refine=False,  # args.with_box_refine,
        two_stage=False,  # args.two_stage,
    )

def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DeformableDETRsegm(
            model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update(
            {k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher,
                             weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, padding=1, kernel_size=5, activation1=None, activation2=None):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.activation1 = activation1
        self.activation2 = activation2

    def forward(self, x):
        out = self.depthwise(x)
        if self.activation1 is not None:
            out = self.activation1(out)
        out = self.pointwise(out)
        if self.activation1 is not None:
            out = self.activation2(out)
        return out


class MaskHeadSmallConvIFC(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims,n_future=3 ,output_dict=None):
        super().__init__()

        # inter_dims = [dim, context_dim // 2, context_dim // 4,
        #               context_dim // 8, context_dim // 16, context_dim // 64, context_dim // 128]
        self.n_future = n_future
        gn = 8

        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(gn, dim)
        self.lay2 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(gn, dim)
        self.lay3 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(gn, dim)
        self.lay4 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(gn, dim)
        self.lay5 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(gn, dim)

        self.lay6 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn6 = torch.nn.GroupNorm(gn, dim)

        self.depth_sep_conv2d = depthwise_separable_conv(
            dim, dim, kernel_size=5, padding=2, activation1=F.relu, activation2=F.relu)

        # half_dim = dim/2
        # self.out_lay_1 = torch.nn.Conv2d(
        #     dim, half_dim, 3, padding=1)
        # self.out_lay_2 = torch.nn.Conv2d(
        #     half_dim, 1, 3, padding=1)  # <- This would be differen

        self.convert_to_weight = MLP(dim, dim, dim, 3)
        # if output_dict is not None:
        #     self.future_pred_layers = build_output_convs(
        #         inter_dims[4], output_dict)
        """ 
        outheads_
            - motion_segmentation: 1x5x200x200   - BxFx1xHxW
        """

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], dim, 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], dim, 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], dim, 1)
        self.adapter4 = torch.nn.Conv2d(fpn_dims[3], dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, seg_memory, fpns, hs):
        x = seg_memory
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        
        cur_fpn = self.adapter1(fpns[-1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        
        cur_fpn = self.adapter2(fpns[-2])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)
        
        cur_fpn = self.adapter3(fpns[-3])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)
        
        cur_fpn = self.adapter4(fpns[-4])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)
        
        T = self.n_future

        x = x.unsqueeze(1).repeat(1, T, 1, 1, 1)
        B, BT, C, H, W = x.shape
        BT = B*T
        L, B, N, C = hs.shape
        
        x = self.depth_sep_conv2d(x.view(B*T, C, H, W)).view(B, T, C, H, W)
        
        w = self.convert_to_weight(hs).permute(1, 0, 2, 3)
        w = w.unsqueeze(1).repeat(1, T, 1, 1, 1)

        mask_logits = F.conv2d(x.view(1, BT*C, H, W),
                               w.reshape(B*T*L*N, C, 1, 1), groups=BT)
        
        
        mask_logits = mask_logits.view(
            B, T, L, N, H, W).permute(2, 0, 3, 1, 4, 5)
        return mask_logits


class MaskHeadSmallConvIFC_V2(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, n_future=3, output_dict=None):
        super().__init__()

        # inter_dims = [dim, context_dim // 2, context_dim // 4,
        #               context_dim // 8, context_dim // 16, context_dim // 64, context_dim // 128]
        self.n_future = n_future
        gn = 8
        T = self.n_future
        fpn_dims = [256,256,512,512]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(gn, dim)
        self.lay2 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(gn, dim)
        self.lay3 = torch.nn.Conv2d(dim, dim*2, 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(gn, dim*2)
        lay4 = torch.nn.Conv2d(dim*2, dim*T, 3, padding=1)
        gn4 = torch.nn.GroupNorm(gn, dim*T)
        lay5 = torch.nn.Conv2d(dim*T, dim*T, 3, padding=1)
        gn5 = torch.nn.GroupNorm(gn, dim*T)

        self.lay6 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn6 = torch.nn.GroupNorm(gn, dim)

        self.depth_sep_conv2d = depthwise_separable_conv(
            dim, dim, kernel_size=5, padding=2, activation1=F.relu, activation2=F.relu)

        # half_dim = dim/2
        # self.out_lay_1 = torch.nn.Conv2d(
        #     dim, half_dim, 3, padding=1)
        # self.out_lay_2 = torch.nn.Conv2d(
        #     half_dim, 1, 3, padding=1)  # <- This would be differen

        self.convert_to_weight = MLP(dim, dim*2, dim*T, 2)
        # if output_dict is not None:
        #     self.future_pred_layers = build_output_convs(
        #         inter_dims[4], output_dict)
        """ 
        outheads_
            - motion_segmentation: 1x5x200x200   - BxFx1xHxW
        """

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], dim, 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], dim, 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], dim, 1)
        self.adapter4 = torch.nn.Conv2d(fpn_dims[3], dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, seg_memory, fpns, hs):
        x = seg_memory
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[-1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[-2])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[-3])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter4(fpns[-4])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        T = self.n_future

        H, W = x.shape[-2:]
        x = x.unsqueeze(1).reshape(1, T, -1, H, W)

        B, BT, C, H, W = x.shape
        L, B, N, C = hs.shape

        x = self.depth_sep_conv2d(x.view(B*BT, C, H, W)).view(B, BT, C, H, W)
        w = self.convert_to_weight(hs).permute(1, 0, 2, 3)
        w = w.unsqueeze(1).reshape(1, T, L, N, -1)
        mask_logits = F.conv2d(x.view(1, BT*C, H, W),
                            w.reshape(B*T*L*N, C, 1, 1), groups=BT)
        mask_logits = mask_logits.view(
            B, T, L, N, H, W).permute(2, 0, 3, 1, 4, 5)

        return mask_logits
    
    
class MaskHeadSmallConvIFC_V3(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, n_future=3, block_future=False, output_dict=None):
        super().__init__()

        # inter_dims = [dim, context_dim // 2, context_dim // 4,
        #               context_dim // 8, context_dim // 16, context_dim // 64, context_dim // 128]
        self.n_future = n_future
        self.block_future = block_future
        gn = 8
        T = self.n_future
        fpn_dims = fpn_dims
        
        self.num_feature_levels = len(fpn_dims)
        
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(gn, dim)
        self.lay2 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(gn, dim)
        self.lay3 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(gn, dim)


        self.a = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=[1, 1, 1], bias=False),
            nn.BatchNorm3d(
                num_features=dim, eps=1e-5, momentum=0.1
            ),
            nn.ReLU(inplace=True),
        )

        # Depthwise (channel-separated) 3x3x3x1 conv
        # Depthwise (channel-separated) 1x3x3x1 spatial conv
        self.b1 = nn.Conv3d(
            dim,
            dim,
            kernel_size=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 1, 1],
            bias=False,
        )
        # Depthwise (channel-separated) 3x1x1x1 temporal conv
        self.b2 = nn.Conv3d(
            dim,
            dim,
            kernel_size=[3, 1, 1],
            stride=[1, 1, 1],
            padding=[1, 0, 0],
            bias=False,
        )
        if self.block_future:
            self.convert_to_weight = MLP(dim, dim, dim, 2)
        else:
            self.convert_to_weight = MLP(dim, dim, dim*T, 2)

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], dim, 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], dim, 1)
        if self.num_feature_levels == 4:
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], dim, 1)
            self.adapter4 = torch.nn.Conv2d(fpn_dims[3], dim*T, 1)
            self.lay3 = torch.nn.Conv2d(dim, dim, 3, padding=1)
            self.gn3 = torch.nn.GroupNorm(gn, dim)
            self.lay4 = torch.nn.Conv2d(dim, dim*T, 3, padding=1)
            self.gn4 = torch.nn.GroupNorm(gn, dim*T)
        elif self.num_feature_levels == 3:
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], dim*T, 1)
            self.lay3 = torch.nn.Conv2d(dim, dim*T, 3, padding=1)
            self.gn3 = torch.nn.GroupNorm(gn, dim*T)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, seg_memory, fpns, hs):
        x = seg_memory # torch.Size([1, 128, 13, 13])  B HD H W 
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        
        cur_fpn = self.adapter1(fpns[-1]) #torch.Size([1, 128, 13, 13])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
        
        cur_fpn = self.adapter2(fpns[-2])  # torch.Size([1, 128, 25, 25])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[-3])  # torch.Size([1, 640, 50, 50]) HD*T
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        
        if self.num_feature_levels == 4:
            x = self.lay4(x)
            x = self.gn4(x)
            x = F.relu(x)

            cur_fpn = self.adapter4(fpns[-4])
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")

        if self.block_future:
            T = self.n_future  # 5
            H, W = x.shape[-2:]
            B = x.shape[0]

            # torch.Size([1, 128, 5, 50, 50])
            x = x.unsqueeze(1).reshape(B, -1, T, H, W)
            # D2+1 Module from Vov3D -> Basically replacing a 3D Convolution
            x = self.b1(x)
            x = self.b2(x)
            x = F.relu(x)
            x = self.a(x).permute(0, 2, 1, 3, 4)

            B, BT, C, H, W = x.shape
            L, B, N, C = hs.shape  # torch.Size([1, 1, 300, 128]) TxBxNxC
            L = 1
            w = self.convert_to_weight(hs).permute(1, 0, 2, 3)
            w = w.unsqueeze(1).reshape(B, T, L, N, -1)
            # torch.Size([1, 5, 1, 300, 128])
            # Unsure about the fusion across the batch dimension - meh IFC segmentation module afterwards
            mask_logits = F.conv2d(x.reshape(1, B * BT*C, H, W),
                                   w.reshape(B*T*L*N, C, 1, 1), groups=BT*B)
            #torch.Size([1, 1500, 50, 50])
            mask_logits = mask_logits.view(
                B, T, L, N, H, W).permute(2, 0, 3, 1, 4, 5)
            #torch.Size([1, 1, 300, 5, 50, 50])

        else:
            T = self.n_future #5
            H, W = x.shape[-2:]
            B = x.shape[0]

            # torch.Size([1, 128, 5, 50, 50])
            x = x.unsqueeze(1).reshape(B, -1, T, H, W)
            # D2+1 Module from Vov3D -> Basically replacing a 3D Convolution 
            x = self.b1(x)
            x = self.b2(x)
            x = F.relu(x)
            x = self.a(x).permute(0, 2, 1, 3, 4)

            B, BT, C, H, W = x.shape
            L, B, N, C = hs.shape  # torch.Size([1, 1, 300, 128]) TxBxNxC
            
            w = self.convert_to_weight(hs).permute(1, 0, 2, 3)
            w = w.unsqueeze(1).reshape(B, T, L, N, -1)
            # torch.Size([1, 5, 1, 300, 128])
            # Unsure about the fusion across the batch dimension - meh IFC segmentation module afterwards
            mask_logits = F.conv2d(x.reshape(1, B *BT*C, H, W),
                                w.reshape(B*T*L*N, C, 1, 1), groups=BT*B)
            #torch.Size([1, 1500, 50, 50])
            mask_logits = mask_logits.view(
                B, T, L, N, H, W).permute(2, 0, 3, 1, 4, 5)
            #torch.Size([1, 1, 300, 5, 50, 50])
        return mask_logits
