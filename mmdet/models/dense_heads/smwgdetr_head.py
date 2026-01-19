# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import InstanceList, reduce_mean, OptInstanceList
from .dino_head import DINOHead

from functools import partial
from ..utils import multi_apply
from .deformable_detr_head import DeformableDETRHead
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps)
from scipy.optimize import linear_sum_assignment

# 这个函数为了调整接口，重写了loss_dn


def parallel_apply(func, input1, input2, input3, input4, input5, **kwargs):
    """
    Apply `func` to each set of (input1[i], input2[i], input3[i])
    and return a list of N (H, W) tensors.

    Args:
        func: Function to be applied. Accepts 3 tensors of shape (H, W), returns (H, W)
        input1, input2, input3: Tensors of shape (N, H, W)

    Returns:
        List of N tensors, each of shape (H, W)
    """
    pfunc = partial(func, **kwargs) if kwargs else func

    # 使用 map 并直接转换为 list
    return list(map(pfunc, input1, input2, input3, input4, input5))


def kld_dis(bbox_pred, bbox_target, mode=None):
    # 假设 bbox_target 和 bbox_pred 的 shape 都是 (N, 4)
    # 其中 4 是 (center_x, center_y, width, height)

    eps = 1e-6  # 防止除零

    # 计算中心差异
    whs = bbox_target[:, :2] - bbox_pred[:, :2]  # (N, 2)

    # 计算宽高
    w_gt = bbox_target[:, 2] + eps  # (N,)
    h_gt = bbox_target[:, 3] + eps  # (N,)
    w_box = bbox_pred[:, 2] + eps  # (N,)
    h_box = bbox_pred[:, 3] + eps  # (N,)

    distance = (w_box ** 2 / w_gt ** 2 + h_box ** 2 / h_gt ** 2 + 4 * whs[..., 0] ** 2 / w_gt ** 2 + 4 * whs[
        ..., 1] ** 2 / h_gt ** 2 + torch.log(w_gt ** 2 / w_box ** 2) + torch.log(
        h_gt ** 2 / h_box ** 2) - 2) / 2

    if not mode:
        score = 1 / (1 + distance)
    elif mode == 'sqrt':
        score = 1 / (1 + torch.sqrt(distance))
    else:
        raise Exception("这是一个错误信息")

    return score


def get_pos_neg_indices(N, num_sample, num_groups):
    assert N % (2 * num_groups) == 0, "N must be divisible by 2 * num_groups"

    group_size = N // (2 * num_groups)
    pos_indices = []
    neg_indices = []

    for g in range(2 * num_groups):
        start = g * group_size
        end = start + group_size
        group_indices = torch.arange(start, end)

        # 取该组前 num_sample 个索引
        selected = group_indices[:num_sample]

        if g % 2 == 0:
            # 偶数组是正组
            pos_indices.append(selected)
        else:
            # 奇数组是负组
            neg_indices.append(selected)

    pos_indices = torch.cat(pos_indices, dim=0)
    neg_indices = torch.cat(neg_indices, dim=0)
    return pos_indices, neg_indices


@MODELS.register_module()
class SMWGDETRHead(DINOHead):
    r"""Head of the DETRs Beat YOLOs on Real-time Object Detection

    Code is modified from the `official github repo
    <https://github.com/PaddlePaddle/PaddleDetection>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2304.08069>`_ .
    """

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor]:
        """Forward function. In RT-DETR, regression and classification are
        performed in the transformer decoder.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        outputs_classes = hidden_states
        outputs_coords = references

        return outputs_classes, outputs_coords

    def DDQS(self, dn_cls, dn_bbox, match_cls, match_bbox,
             dn_cls_last, dn_bbox_last, match_cls_last, match_bbox_last,
             bbox_target, labels, bbox_weight, label_weight, num_groups=1):

        bbox_target = bbox_target.clone()  # 防止原target张量被修改
        labels = labels.clone()

        if bbox_target.size()[0] == 0:  # 没有GT输入时直接原样输出
            return label_weight, bbox_weight, labels, bbox_target
        if torch.all(dn_bbox == 0):  # 全零的话输出，用于控制last
            return label_weight, bbox_weight, labels, bbox_target

        if self.mode >= 10:  # mode>10时，说明使用上一层结果进行判断
            if torch.all(dn_bbox_last == 0):  # 全零的话输出，用于控制last
                return label_weight, bbox_weight, labels, bbox_target

            dn_cls_cau, dn_bbox_cau, match_cls_cau, match_bbox_cau = dn_cls_last, dn_bbox_last, match_cls_last, match_bbox_last
        else:
            dn_cls_cau, dn_bbox_cau, match_cls_cau, match_bbox_cau = dn_cls, dn_bbox, match_cls, match_bbox

        # 筛选正负样本id
        mask = (bbox_target != 0).any(dim=1)
        pos_index = mask.nonzero(as_tuple=True)[0]

        num_sample = int(len(pos_index) / num_groups)  # 每组里面正样本的个数

        pos_bbox_index, neg_bbox_index = get_pos_neg_indices(dn_bbox.size(0), int(len(pos_index) / num_groups),
                                                             num_groups)
        pos_bbox_index, neg_bbox_index = (pos_bbox_index.to(dn_bbox.device).view(num_groups, num_sample),
                                          neg_bbox_index.to(dn_bbox.device).view(num_groups, num_sample))

        # 根据id取数据
        dn_bbox_pos = dn_bbox_cau[pos_bbox_index]  # (num_group, num_sample, 4)
        dn_cls_pos = dn_cls_cau[pos_bbox_index]

        dn_bbox_neg = dn_bbox_cau[neg_bbox_index]
        dn_cls_neg = dn_cls_cau[neg_bbox_index]

        bbox_target_cau = bbox_target[pos_index]
        bbox_target_cau = bbox_target_cau[0: int(bbox_target_cau.size()[0] / num_groups)]

        labels_cau = labels[pos_index]
        labels_cau = labels_cau[0: int(labels_cau.size()[0] / num_groups)]

        if self.mode == 10:  # 只修改正样本对应的GT
            bbox_all = torch.cat([dn_bbox_pos.view(-1, 4), dn_bbox_neg.view(-1, 4), match_bbox_cau], dim=0)
            cls_all = torch.cat([dn_cls_pos.view(-1, 8), dn_cls_neg.view(-1, 8), match_cls_cau], dim=0)

            gt_bboxes = bbox_target_cau
            pred_bboxes = bbox_all

            cost_1 = torch.cdist(pred_bboxes, gt_bboxes, p=1) * 5

            gt_bboxes = bbox_cxcywh_to_xyxy(gt_bboxes) * 800
            pred_bboxes = bbox_cxcywh_to_xyxy(bbox_all) * 800

            overlaps = bbox_overlaps(
                pred_bboxes, gt_bboxes, mode='giou', is_aligned=False)

            iou_cost = -overlaps
            cost_2 = iou_cost * 2

            cls_pred = cls_all.sigmoid()
            neg_cost = -(1 - cls_pred + 1e-12).log() * (1 - 0.25) * cls_pred.pow(2)
            pos_cost = -(cls_pred + 1e-12).log() * 0.25 * (1 - cls_pred).pow(2)

            cls_cost = pos_cost[:, labels_cau] - neg_cost[:, labels_cau]
            cost_3 = cls_cost * 2

            cost = cost_1 + cost_2 + cost_3

            cost_denoise_pos = cost[0: num_groups * num_sample].view(num_groups, num_sample, num_sample)
            cost_denoise_neg = cost[num_groups * num_sample: 2 * num_groups * num_sample].view(num_groups, num_sample,
                                                                                               num_sample)
            cost_match = cost[2 * num_groups * num_sample:]

            _, cost_match_each_gt = self.hungarian(cost_match.cpu())

            cost_match_each_gt = cost_match_each_gt.to(cost_match.device)
            cost_match_each_gt = cost_match_each_gt.unsqueeze(0).repeat(num_sample, 1).unsqueeze(0).repeat(num_groups,
                                                                                                           1, 1)

            index_judge = torch.any((cost_match_each_gt >= cost_denoise_neg), dim=-1).flatten()
            remove_index = neg_bbox_index.flatten()[index_judge]

            label_weight[remove_index] = 0

            if (cost_denoise_pos.size()[0] == 0) | (cost_denoise_pos.size()[1] == 0):
                return label_weight, bbox_weight, labels, bbox_target

            cost_com = cost_denoise_pos.min(1)[0].max(0)[0].unsqueeze(0).unsqueeze(0).repeat(num_groups, num_sample, 1)

            index_judge = torch.any((cost_com >= cost_denoise_neg), dim=-1).flatten()
            remove_index = neg_bbox_index.flatten()[index_judge]

            label_weight[remove_index] = 0

            return label_weight, bbox_weight, labels, bbox_target

        return label_weight, bbox_weight

    def hungarian(self, cost):
        """
        Args:
            cost: (num_queries, num_gts) 的代价矩阵

        Returns:
            assigned_gt_inds: 每个 query 分配的 GT 索引（从 1 开始，0 表示未分配，-1 表示无效）
            matched_costs_sorted: 按照 GT id（1 ~ n）升序排列的匹配 cost 值，形状为 (num_gts,)，
                                  未被匹配的 GT 对应位置为 inf。
        """
        device = cost.device
        num_queries, num_gts = cost.shape
        assigned_gt_inds = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        matched_costs_sorted = torch.full((num_gts,), float('inf'), device=device)  # 初始化为 inf

        # 匈牙利匹配
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost.cpu().numpy())
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 分配 query 到 GT
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1  # +1 使得 GT index 从 1 开始

        # 计算匹配 cost，并放到对应 GT 的位置上
        matched_costs = cost[matched_row_inds, matched_col_inds]  # shape = (num_matched,)
        matched_costs_sorted[matched_col_inds] = matched_costs  # 将每个 GT 的 cost 放在对应位置

        return assigned_gt_inds, matched_costs_sorted

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None,
        batch_data_samples = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        loss_dict = super(DeformableDETRHead, self).loss_by_feat(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                all_layers_matching_cls_scores,
                all_layers_matching_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta,
                batch_data_samples=batch_data_samples
            )
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                all_layers_matching_cls_scores,
                all_layers_matching_bbox_preds,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int], batch_data_samples) -> Tuple[List[Tensor]]:
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        denoise_bbox_last_hold = all_layers_denoising_bbox_preds.detach()
        denoise_bbox_last = torch.zeros_like(denoise_bbox_last_hold)
        denoise_bbox_last[1:] = denoise_bbox_last_hold[0: -1]

        denoise_cls_last_hold = all_layers_denoising_cls_scores.detach()
        denoise_cls_last = torch.zeros_like(denoise_cls_last_hold)
        denoise_cls_last[1:] = denoise_cls_last_hold[0: -1]

        match_bbox_last_hold = all_layers_matching_bbox_preds.detach()
        match_bbox_last = torch.zeros_like(match_bbox_last_hold)
        match_bbox_last[1:] = match_bbox_last_hold[0: -1]

        match_cls_last_hold = all_layers_matching_cls_scores.detach()
        match_cls_last = torch.zeros_like(match_cls_last_hold)
        match_cls_last[1:] = match_cls_last_hold[0: -1]

        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            denoise_cls_last,
            denoise_bbox_last,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            match_cls_last_hold,
            match_bbox_last,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta,
            batch_data_samples=batch_data_samples
        )

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        dn_cls_last: Tensor, dn_bbox_last: Tensor,
                        matching_cls_scores,
                        matching_bbox_preds,
                        match_cls_last,
                        match_bbox_last,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int], batch_data_samples) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:9
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta, batch_data_samples)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # 框权重调制
        if self.weighted_denoise == 1:
            label_weights_list, bbox_weights_list, labels_list, bbox_targets_list = multi_apply(
                self.DDQS,
                dn_cls_scores.detach(),
                dn_bbox_preds.detach(),
                matching_cls_scores.detach(),
                matching_bbox_preds.detach(),
                dn_cls_last.detach(),
                dn_bbox_last.detach(),
                match_cls_last.detach(),
                match_bbox_last.detach(),
                bbox_targets_list,
                labels_list,
                bbox_weights_list,
                label_weights_list,
                num_groups=dn_meta['num_denoising_groups']
            )

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors  # 预测框与GT框一对一
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        if self.loss_cls.__class__.__name__ != 'MalLoss':
            iou_score = bbox_overlaps(
                bboxes.detach(), bboxes_gt, is_aligned=True).clamp(min=1e-6)  # VaL计算的是IOU得分
        else:
            if self.loss_cls.score_type == 'iou':
                iou_score = bbox_overlaps(
                    bboxes.detach(), bboxes_gt, is_aligned=True).clamp(min=1e-6)
            elif self.loss_cls.score_type == 'kld_sqrt':
                iou_score = kld_dis(bbox_preds.detach(), bbox_targets, 'sqrt')
            elif self.loss_cls.score_type == 'kld':
                iou_score = kld_dis(bbox_preds.detach(), bbox_targets)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        cls_iou_targets = torch.zeros_like(cls_scores)
        one_hot_tagets = torch.zeros_like(cls_iou_targets)
        if labels.numel() > 0:
            # valid indices of classification targets
            valid_idx = labels < self.cls_out_channels
            # assign iou score to the corresponding label
            cls_iou_targets[valid_idx, labels[valid_idx]] = iou_score[valid_idx]
            one_hot_tagets[valid_idx, labels[valid_idx]] = 1

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if self.loss_cls.__class__.__name__ == 'MalLoss':
                loss_cls = self.loss_cls(
                    cls_scores, cls_iou_targets, one_hot_tagets, None, avg_factor=cls_avg_factor)
            else:
                loss_cls = self.loss_cls(
                    cls_scores, cls_iou_targets, label_weights.unsqueeze(1).repeat(1, self.num_classes), avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        if self.loss_cls.__class__.__name__ != 'MalLoss':
            iou_score = bbox_overlaps(
                bboxes.detach(), bboxes_gt, is_aligned=True).clamp(min=1e-6)  # VaL计算的是IOU得分
        else:
            if self.loss_cls.score_type == 'iou':
                iou_score = bbox_overlaps(
                    bboxes.detach(), bboxes_gt, is_aligned=True).clamp(min=1e-6)
            elif self.loss_cls.score_type == 'kld_sqrt':
                iou_score = kld_dis(bbox_preds.detach(), bbox_targets, 'sqrt')
            elif self.loss_cls.score_type == 'kld':
                iou_score = kld_dis(bbox_preds.detach(), bbox_targets)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_iou_targets = torch.zeros_like(cls_scores)
        one_hot_tagets = torch.zeros_like(cls_iou_targets)
        if labels.numel() > 0:
            # valid indices of classification targets
            valid_idx = labels < self.cls_out_channels
            # assign iou score to the corresponding label
            cls_iou_targets[valid_idx, labels[valid_idx]] = iou_score[valid_idx]
            one_hot_tagets[valid_idx, labels[valid_idx]] = 1

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight

        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if self.loss_cls.__class__.__name__ == 'MalLoss':
            loss_cls = self.loss_cls(
                cls_scores, cls_iou_targets, one_hot_tagets, None, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, cls_iou_targets, None, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou
