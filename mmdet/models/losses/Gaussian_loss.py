# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps
from .utils import weighted_loss
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh


def cau_gaussian_cost(gt, box, tau, f_x):
    eps = 1e-10
    center_gt = gt[:, :2]
    center_box = box[:, :2]
    whs = center_gt - center_box

    w_gt = gt[:, 2] + eps
    h_gt = gt[:, 3] + eps
    w_box = box[:, 2] + eps
    h_box = box[:, 3] + eps

    distance = (w_box ** 2 / w_gt ** 2 + h_box ** 2 / h_gt ** 2 + 4 * whs[..., 0] ** 2 / w_gt ** 2 + 4 * whs[
        ..., 1] ** 2 / h_gt ** 2 +
                torch.log(w_gt ** 2 / w_box ** 2) + torch.log(h_gt ** 2 / h_box ** 2) - 2) / 2

    if f_x == 'x':
        kl_distance = 1 / (tau + distance)  # KL散度距离
    elif f_x == 'ln':
        kl_distance = 1 / (tau + torch.log1p(distance))
    elif f_x == 'sqrt':
        kl_distance = 1 / (tau + torch.sqrt(distance))
    else:
        raise ValueError('输入的变化函数不是可选的类型')

    kl_cost = 1 - kl_distance * tau

    return kl_cost


@weighted_loss
def gaussian_loss(pred: Tensor, target: Tensor, eps: float = 1e-7, tau=1, f_x='x') -> Tensor:
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    if (target > 1).any():  # xyxy格式的
        pred = bbox_xyxy_to_cxcywh(pred) / 800
        target = bbox_xyxy_to_cxcywh(target) / 800

    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    distance = cau_gaussian_cost(target, pred, tau, f_x).t()

    if fp16:
        distance = distance.to(torch.float16)

    loss = distance

    return loss


@MODELS.register_module()
class GaussianKldLoss(nn.Module):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 tau=1,
                 f_x='x'
                 ) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.tau = tau
        self.f_x = f_x

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * gaussian_loss(
            pred,
            target,
            weight,
            tau=self.tau,
            f_x=self.f_x,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss



