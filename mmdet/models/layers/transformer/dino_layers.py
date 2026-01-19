# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Tuple, Union

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn
import random

from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.utils import OptConfigType
from .deformable_detr_layers import DeformableDetrTransformerDecoder
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid
import time
from functools import partial
from mmdet.structures.bbox import bbox_overlaps


def multi_apply_mine(func, a, b, **kwargs):
    """Apply function to each pair (a[i], b[i]) and return three lists of results.

    Args:
        func (Function): A function that takes inputs (a_i, b_i) and returns
            a tuple of three results (e.g. x_i, y_i, z_i).
        a (list): A list with length n, each element a_i can have different lengths.
        b (Tensor): A tensor of shape (n, m).

    Returns:
        tuple: Three lists, each of length n.
               The i-th element of each list corresponds to one of the values returned by func(a_i, b_i).
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = list(map(pfunc, a, b))  # [(x_0, y_0, z_0), (x_1, y_1, z_1), ...]

    # 解压结果，将 map_results 列表中每个元组对应的元素分别解出来
    # zip(*map_results) 会将 [(x_0, y_0, z_0), (x_1, y_1, z_1), ...]
    # 解压为 [x_0, x_1, ...], [y_0, y_1, ...], [z_0, z_1, ...]
    results = list(zip(*map_results))

    # 此时 results 是一个包含三个元组的元组 ( (x_0, x_1, ...), (y_0, y_1, ...), (z_0, z_1, ...) )
    # 如果需要将其变为list类型，可以这样:
    return [list(r) for r in results]


def cal_gussian_dis(gt, box, num_groups=1, mode='kld'):
    """
    输入的gt是xyxy格式
    init_box是xywh格式
    """

    gt = bbox_xyxy_to_cxcywh(gt)
    eps = 1e-10
    if mode == 'wd':

        center_gt = gt[:, None, :2]
        center_box = box[None, :, :2]
        whs = center_gt - center_box
        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps

        w_gt = gt[:, None, 2] + eps
        h_gt = gt[:, None, 3] + eps
        w_box = box[None, :, 2] + eps
        h_box = box[None, :, 3] + eps
        wh_distance = ((w_gt - w_box) ** 2 + (h_gt - h_box) ** 2) / 4

        distance = (center_distance + wh_distance) * 800

        wd_distance = 1 / (1 + distance)

        return wd_distance

    elif mode == 'kld':  # 这一块KL距离是box到gt的距离，后面只要是按着这个函数计算应该就没问题

        center_gt = gt[:, None, :2]
        center_box = box[None, :, :2]
        whs = center_gt - center_box

        w_gt = gt[:, None, 2] + eps
        h_gt = gt[:, None, 3] + eps
        w_box = box[None, :, 2] + eps
        h_box = box[None, :, 3] + eps

        distance = (w_box ** 2 / w_gt ** 2 + h_box ** 2 / h_gt ** 2 + 4 * whs[..., 0] ** 2 / w_gt ** 2 + 4 * whs[
            ..., 1] ** 2 / h_gt ** 2 + torch.log(w_gt ** 2 / w_box ** 2) + torch.log(h_gt ** 2 / h_box ** 2) - 2) / 2

        kl_distance = 1 / (1 + distance)  # KL散度距离

        device = kl_distance.device
        num_gt, num_box = kl_distance.shape
        sorted_scores, sorted_indices = torch.sort(kl_distance, dim=1, descending=True)

        # 初始化结果张量
        # final_scores[i]: 分配给第i个GT的最终得分
        # final_box[i]: 分配给第i个GT的box索引
        final_scores = torch.full((num_gt,), -1.0, device=device)
        final_box = torch.full((num_gt,), -1, dtype=torch.long, device=device)

        # assigned_gt_for_box[j]: 当前被box j分配的GT编号，-1表示box空闲
        assigned_gt_for_box = torch.full((num_box,), -1, dtype=torch.long, device=device)

        # # 我们需要一种机制让被踢出的GT能重新尝试下一个box
        # next_try_idx[i]: 下一个该GT尝试的box在sorted_indices[i]中的下标（从0开始）
        next_try_idx = torch.zeros(num_gt, dtype=torch.long, device=device)

        # 将所有GT入队列(使用列表模拟)，初始时所有GT都等待匹配
        gt_queue = torch.arange(num_gt, device=device).tolist()

        while gt_queue:
            gt = gt_queue.pop(0)
            t_idx = next_try_idx[gt].item()

            if t_idx >= num_box:
                continue

            # 当前GT尝试下一个box
            box_candidate = sorted_indices[gt, t_idx].item()
            cand_score = sorted_scores[gt, t_idx].item()

            # 如果该box空闲，直接分配
            current_gt_for_box = assigned_gt_for_box[box_candidate].item()
            if current_gt_for_box == -1:
                assigned_gt_for_box[box_candidate] = gt
                final_scores[gt] = cand_score
                final_box[gt] = box_candidate
            else:
                # box已被占用，比较得分
                current_gt_score_on_this_box = kl_distance[current_gt_for_box, box_candidate].item()
                if cand_score > current_gt_score_on_this_box:
                    # 当前GT在该box上的得分更高，踢出现有占用者
                    assigned_gt_for_box[box_candidate] = gt
                    final_scores[gt] = cand_score
                    final_box[gt] = box_candidate

                    # 被踢出的GT需要重新尝试下一个box
                    # 首先清空其分配记录
                    final_scores[current_gt_for_box] = -1.0
                    final_box[current_gt_for_box] = -1
                    next_try_idx[current_gt_for_box] += 1
                    if next_try_idx[current_gt_for_box].item() < num_box:
                        # 如果该GT仍有box可尝试，将其重新加入队列
                        gt_queue.append(current_gt_for_box)
                else:
                    # 当前GT在该box上的得分不如已分配的GT，则GT尝试下一个box
                    next_try_idx[gt] += 1
                    if next_try_idx[gt].item() < num_box:
                        gt_queue.append(gt)
        kl_distance[:, final_box] = 0  # 用于筛选负样本的
        neg_scores, _ = torch.topk(kl_distance, num_groups, dim=1)
        # neg_scores = torch.mean(neg_scores, dim=1, keepdim=True)
        # neg_scores = neg_scores.repeat(1, num_groups)

        return final_scores, neg_scores

    elif mode == 'giou':  # 这一块KL距离是box到gt的距离，后面只要是按着这个函数计算应该就没问题

        distance_bbox = torch.cdist(gt, box, p=1) * 5

        gt = bbox_cxcywh_to_xyxy(gt) * 800
        box = bbox_cxcywh_to_xyxy(box) * 800
        distance_giou = (1 - bbox_overlaps(gt, box, mode='giou', is_aligned=False)) * 2

        distance = distance_bbox + distance_giou

        device = distance.device
        num_gt, num_box = distance.shape
        sorted_scores, sorted_indices = torch.sort(distance, dim=1, descending=True)

        # 初始化结果张量
        # final_scores[i]: 分配给第i个GT的最终得分
        # final_box[i]: 分配给第i个GT的box索引
        final_scores = torch.full((num_gt,), -1.0, device=device)
        final_box = torch.full((num_gt,), -1, dtype=torch.long, device=device)

        # assigned_gt_for_box[j]: 当前被box j分配的GT编号，-1表示box空闲
        assigned_gt_for_box = torch.full((num_box,), -1, dtype=torch.long, device=device)

        # # 我们需要一种机制让被踢出的GT能重新尝试下一个box
        # next_try_idx[i]: 下一个该GT尝试的box在sorted_indices[i]中的下标（从0开始）
        next_try_idx = torch.zeros(num_gt, dtype=torch.long, device=device)

        # 将所有GT入队列(使用列表模拟)，初始时所有GT都等待匹配
        gt_queue = torch.arange(num_gt, device=device).tolist()

        while gt_queue:
            gt = gt_queue.pop(0)
            t_idx = next_try_idx[gt].item()

            if t_idx >= num_box:
                continue

            # 当前GT尝试下一个box
            box_candidate = sorted_indices[gt, t_idx].item()
            cand_score = sorted_scores[gt, t_idx].item()

            # 如果该box空闲，直接分配
            current_gt_for_box = assigned_gt_for_box[box_candidate].item()
            if current_gt_for_box == -1:
                assigned_gt_for_box[box_candidate] = gt
                final_scores[gt] = cand_score
                final_box[gt] = box_candidate
            else:
                # box已被占用，比较得分
                current_gt_score_on_this_box = distance[current_gt_for_box, box_candidate].item()
                if cand_score > current_gt_score_on_this_box:
                    # 当前GT在该box上的得分更高，踢出现有占用者
                    assigned_gt_for_box[box_candidate] = gt
                    final_scores[gt] = cand_score
                    final_box[gt] = box_candidate

                    # 被踢出的GT需要重新尝试下一个box
                    # 首先清空其分配记录
                    final_scores[current_gt_for_box] = -1.0
                    final_box[current_gt_for_box] = -1
                    next_try_idx[current_gt_for_box] += 1
                    if next_try_idx[current_gt_for_box].item() < num_box:
                        # 如果该GT仍有box可尝试，将其重新加入队列
                        gt_queue.append(current_gt_for_box)
                else:
                    # 当前GT在该box上的得分不如已分配的GT，则GT尝试下一个box
                    next_try_idx[gt] += 1
                    if next_try_idx[gt].item() < num_box:
                        gt_queue.append(gt)

        distance[:, final_box] = 0  # 用于筛选负样本的
        neg_scores, neg_indices = torch.topk(distance, num_groups, dim=1)

        return final_scores, neg_scores


class DinoTransformerDecoder(DeformableDetrTransformerDecoder):
    """Transformer decoder of DINO."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super()._init_layers()
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []  # 这个单词的意思是中间状态
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


def parallel_apply(func, input1, input2, input3, **kwargs):

    pfunc = partial(func, **kwargs) if kwargs else func

    out_list = list(map(pfunc, input1, input2, input3))

    return torch.stack(out_list, dim=0)


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


def cau_dis(gt, bbox):
    eps = 1e-10
    center_gt = gt[:, None, :2]
    center_box = bbox[None, :, :2]
    whs = center_gt - center_box

    w_gt = gt[:, None, 2] + eps
    h_gt = gt[:, None, 3] + eps
    w_box = bbox[None, :, 2] + eps
    h_box = bbox[None, :, 3] + eps

    distance = (w_box ** 2 / w_gt ** 2 + h_box ** 2 / h_gt ** 2 + 4 * whs[..., 0] ** 2 / w_gt ** 2 + 4 * whs[
        ..., 1] ** 2 / h_gt ** 2 + torch.log(w_gt ** 2 / w_box ** 2) + torch.log(
        h_gt ** 2 / h_box ** 2) - 2) / 2

    kld_score = 1 / (1 + torch.sqrt(distance)).T  # KL散度距离

    return kld_score


class CdnQueryGenerator(BaseModule):
    """Implement query generator of the Contrastive denoising (CDN) proposed in
    `DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object
    Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        num_classes (int): Number of object classes.
        embed_dims (int): The embedding dimensions of the generated queries.
        num_matching_queries (int): The queries number of the matching part.
            Used for generating dn_mask.
        label_noise_scale (float): The scale of label noise, defaults to 0.5.
        box_noise_scale (float): The scale of box noise, defaults to 1.0.
        group_cfg (:obj:`ConfigDict` or dict, optional): The config of the
            denoising queries grouping, includes `dynamic`, `num_dn_queries`,
            and `num_groups`. Two grouping strategies, 'static dn groups' and
            'dynamic dn groups', are supported. When `dynamic` is `False`,
            the `num_groups` should be set, and the number of denoising query
            groups will always be `num_groups`. When `dynamic` is `True`, the
            `num_dn_queries` should be set, and the group number will be
            dynamic to ensure that the denoising queries number will not exceed
            `num_dn_queries` to prevent large fluctuations of memory. Defaults
            to `None`.
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,  # 原配置是1.0
                 group_cfg: OptConfigType = None,
                 gaussian_denoise: bool = False,  # 是否启用高斯去噪
                 gaussian_denoise_config=None,
                 max_gt_bbox=500,
                 remove_denoise=False
                 ) -> None:
        super().__init__()

        self.gaussian_denoise = gaussian_denoise
        self.gaussian_denoise_config = gaussian_denoise_config
        self.max_gt_bbox = max_gt_bbox

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale
        self.remove_denoise = remove_denoise
        #
        self.num_iter = 0

        # prepare grouping strategy
        group_cfg = {} if group_cfg is None else group_cfg
        self.dynamic_dn_groups = group_cfg.get('dynamic', True)
        if self.dynamic_dn_groups:
            if 'num_dn_queries' not in group_cfg:
                warnings.warn("'num_dn_queries' should be set when using "
                              'dynamic dn groups, use 100 as default.')
            self.num_dn_queries = group_cfg.get('num_dn_queries', 100)
            assert isinstance(self.num_dn_queries, int), \
                f'Expected the num_dn_queries to have type int, but got ' \
                f'{self.num_dn_queries}({type(self.num_dn_queries)}). '
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using static dn groups'
            self.num_groups = group_cfg['num_groups']
            assert isinstance(self.num_groups, int), \
                f'Expected the num_groups to have type int, but got ' \
                f'{self.num_groups}({type(self.num_groups)}). '

        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates `Unknown` class. However, the embedding of `unknown` class
        # is not used in the original DINO.
        # TODO: num_classes + 1 or num_classes ?
        self.label_embedding = nn.Embedding(self.num_classes, self.embed_dims)

    def __call__(self, batch_data_samples: SampleList,
                 init_boxes=None, init_scores=None,
                 num_match_query=None) -> tuple:
        """Generate contrastive denoising (cdn) queries with ground truth.

        Descriptions of the Number Values in code and comments:
            - num_target_total: the total target number of the input batch
              samples.
            - max_num_target: the max target number of the input batch samples.
            - num_noisy_targets: the total targets number after adding noise,
              i.e., num_target_total * num_groups * 2.
            - num_denoising_queries: the length of the output batched queries,
              i.e., max_num_target * num_groups * 2.

        NOTE The format of input bboxes in batch_data_samples is unnormalized
        (x, y, x, y), and the output bbox queries are embedded by normalized
        (cx, cy, w, h) format bboxes going through inverse_sigmoid.

        Args:
            batch_data_samples (list[:obj:`DetDataSample`]): List of the batch
                data samples, each includes `gt_instance` which has attributes
                `bboxes` and `labels`. The `bboxes` has unnormalized coordinate
                format (x, y, x, y).

        Returns:
            tuple: The outputs of the dn query generator.

            - dn_label_query (Tensor): The output content queries for denoising
              part, has shape (bs, num_denoising_queries, dim), where
              `num_denoising_queries = max_num_target * num_groups * 2`.
            - dn_bbox_query (Tensor): The output reference bboxes as positions
              of queries for denoising part, which are embedded by normalized
              (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
              shape (bs, num_denoising_queries, 4) with the last dimension
              arranged as (cx, cy, w, h).
            - attn_mask (Tensor): The attention mask to prevent information
              leakage from different denoising groups and matching parts,
              will be used as `self_attn_mask` of the `decoder`, has shape
              (num_queries_total, num_queries_total), where `num_queries_total`
              is the sum of `num_denoising_queries` and `num_matching_queries`.
            - dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.
        """
        # normalize bbox and collate ground truth (gt)
        if num_match_query is not None:
            self.num_matching_queries = num_match_query  # 可变初始化查询的时候用
        gt_labels_list = []
        gt_bboxes_list = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            # 为了限制最大框数加的代码
            if sample.gt_instances.bboxes.size(0) > self.max_gt_bbox:
                bboxes = sample.gt_instances.bboxes[sample.selected_gt]
                factor = bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
                bboxes_normalized = bboxes / factor
                gt_bboxes_list.append(bboxes_normalized)
                gt_labels_list.append(sample.gt_instances.labels[sample.selected_gt])
            else:
                bboxes = sample.gt_instances.bboxes
                factor = bboxes.new_tensor([img_w, img_h, img_w,
                                            img_h]).unsqueeze(0)
                bboxes_normalized = bboxes / factor
                gt_bboxes_list.append(bboxes_normalized)
                gt_labels_list.append(sample.gt_instances.labels)

        gt_labels = torch.cat(gt_labels_list)  # (num_target_total, 4)
        gt_bboxes = torch.cat(gt_bboxes_list)

        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)
        self.num_groups = num_groups

        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)

        if self.gaussian_denoise is not True:
            dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)  # 原本的加噪策略
        else:
            pass

        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        batch_idx = torch.cat([
            torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)
        ])

        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, batch_idx, len(batch_data_samples),
            num_groups)

        if self.remove_denoise:
            print("-----------------remove---------------------------")
            dn_bbox_query = parallel_apply(self.remove_false_sample, dn_bbox_query, gt_bboxes_list, num_target_list)


        attn_mask = self.generate_dn_mask(
            max_num_target, num_groups, device=dn_label_query.device)

        dn_meta = dict(
            num_denoising_queries=int(max_num_target * 2 * num_groups),
            num_denoising_groups=num_groups)

        return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def remove_false_sample(self, bbox, gt, num_sample):

        if (bbox.size()[0] > 0) & (gt.size()[0] > 1):
            bbox = bbox.sigmoid()
            gt = bbox_xyxy_to_cxcywh(gt)

            pos_bbox_index, neg_bbox_index = get_pos_neg_indices(bbox.size(0), num_sample, self.num_groups)
            pos_bbox_index, neg_bbox_index = pos_bbox_index.to(gt.device), neg_bbox_index.to(gt.device)

            pos_score = cau_dis(gt, bbox[pos_bbox_index])
            neg_score = cau_dis(gt, bbox[neg_bbox_index])

            pos_remove_index = (pos_score.max(dim=1)[1] != torch.arange(len(pos_score),device=pos_score.device) % num_sample) | \
                               ((pos_score.topk(2, dim=1)[0][:, 0] - pos_score.topk(2, dim=1)[0][:, 1]) <= 0.1)
            pos_remove_index = torch.nonzero(pos_remove_index, as_tuple=False).squeeze().to(gt.device)

            pos_remove_index_abs = pos_bbox_index[pos_remove_index]
            pos_remove_content = gt[pos_remove_index % num_sample]
            if len(pos_remove_content.size()) == 1:
                pos_remove_content = pos_remove_content.unsqueeze(0)

            # 加噪
            if pos_remove_content.size()[0] >= 1:
                rand_sign = torch.randint_like(pos_remove_content, low=0, high=2, dtype=torch.float32, device=gt.device) * 2.0 - 1.0
                rand_part = torch.rand_like(pos_remove_content, device=gt.device) * 0.15
                rand_part *= rand_sign

                bboxes_whwh = pos_remove_content[:, 2:].repeat(1, 2)
                bboxes_xy = bbox_cxcywh_to_xyxy(pos_remove_content)
                pos_remove_content = bboxes_xy + torch.mul(rand_part, bboxes_whwh)
                pos_remove_content = pos_remove_content.clamp(min=0.0, max=1.0)
                pos_remove_content = bbox_xyxy_to_cxcywh(pos_remove_content)

            neg_remove_index = (neg_score.max(dim=1)[1] != torch.arange(len(neg_score),device=neg_score.device) % num_sample) | \
                               ((neg_score.topk(2, dim=1)[0][:, 0] - neg_score.topk(2, dim=1)[0][:, 1]) <= 0.1)
            neg_remove_index = torch.nonzero(neg_remove_index, as_tuple=False).squeeze().to(gt.device)
            neg_remove_index_abs = neg_bbox_index[neg_remove_index]

            bbox[pos_remove_index_abs] = pos_remove_content
            bbox[neg_remove_index_abs] = torch.tensor([0.5, 0.5, 0.5, 0.5], device=gt.device)

            bbox = inverse_sigmoid(bbox, eps=1e-3)

        return bbox

    def get_num_groups(self, max_num_target: int = None) -> int:
        """Calculate denoising query groups number.

        Two grouping strategies, 'static dn groups' and 'dynamic dn groups',
        are supported. When `self.dynamic_dn_groups` is `False`, the number
        of denoising query groups will always be `self.num_groups`. When
        `self.dynamic_dn_groups` is `True`, the group number will be dynamic,
        ensuring the denoising queries number will not exceed
        `self.num_dn_queries` to prevent large fluctuations of memory.

        NOTE The `num_group` is shared for different samples in a batch. When
        the target numbers in the samples varies, the denoising queries of the
        samples containing fewer targets are padded to the max length.

        Args:
            max_num_target (int, optional): The max target number of the batch
                samples. It will only be used when `self.dynamic_dn_groups` is
                `True`. Defaults to `None`.

        Returns:
            int: The denoising group number of the current batch.
        """
        if self.dynamic_dn_groups:
            assert max_num_target is not None, \
                'group_queries should be provided when using ' \
                'dynamic dn groups'
            if max_num_target == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn_queries // max_num_target
        else:
            num_groups = self.num_groups
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def generate_dn_label_query(self, gt_labels: Tensor,
                                num_groups: int) -> Tensor:
        """Generate noisy labels and their query embeddings.

        The strategy for generating noisy labels is: Randomly choose labels of
        `self.label_noise_scale * 0.5` proportion and override each of them
        with a random object category label.

        NOTE Not add noise to all labels. Besides, the `self.label_noise_scale
        * 0.5` arg is the ratio of the chosen positions, which is higher than
        the actual proportion of noisy labels, because the labels to override
        may be correct. And the gap becomes larger as the number of target
        categories decreases. The users should notice this and modify the scale
        arg or the corresponding logic according to specific dataset.

        Args:
            gt_labels (Tensor): The concatenated gt labels of all samples
                in the batch, has shape (num_target_total, ) where
                `num_target_total = sum(num_target_list)`.
            num_groups (int): The number of denoising query groups.

        Returns:
            Tensor: The query embeddings of noisy labels, has shape
            (num_noisy_targets, embed_dims), where `num_noisy_targets =
            num_target_total * num_groups * 2`.
        """
        assert self.label_noise_scale > 0
        gt_labels_expand = gt_labels.repeat(2 * num_groups,
                                            1).view(-1)  # Note `* 2`  # noqa
        p = torch.rand_like(gt_labels_expand.float())
        chosen_indice = torch.nonzero(p < (self.label_noise_scale * 0.5)).view(
            -1)  # Note `* 0.5`
        new_labels = torch.randint_like(chosen_indice, 0, self.num_classes)
        noisy_labels_expand = gt_labels_expand.scatter(0, chosen_indice,
                                                       new_labels)
        dn_label_query = self.label_embedding(noisy_labels_expand)
        return dn_label_query

    def generate_dn_bbox_query(self, gt_bboxes: Tensor,
                               num_groups: int) -> Tensor:
        """Generate noisy bboxes and their query embeddings.

        The strategy for generating noisy bboxes is as follow:

        .. code:: text

            +--------------------+
            |      negative      |
            |    +----------+    |
            |    | positive |    |
            |    |    +-----|----+------------+
            |    |    |     |    |            |
            |    +----+-----+    |            |
            |         |          |            |
            +---------+----------+            |
                      |                       |
                      |        gt bbox        |
                      |                       |
                      |             +---------+----------+
                      |             |         |          |
                      |             |    +----+-----+    |
                      |             |    |    |     |    |
                      +-------------|--- +----+     |    |
                                    |    | positive |    |
                                    |    +----------+    |
                                    |      negative      |
                                    +--------------------+

         The random noise is added to the top-left and down-right point
         positions, hence, normalized (x, y, x, y) format of bboxes are
         required. The noisy bboxes of positive queries have the points
         both within the inner square, while those of negative queries
         have the points both between the inner and outer squares.

        Besides, the length of outer square is twice as long as that of
        the inner square, i.e., self.box_noise_scale * w_or_h / 2.
        NOTE The noise is added to all the bboxes. Moreover, there is still
        unconsidered case when one point is within the positive square and
        the others is between the inner and outer squares.

        Args:
            gt_bboxes (Tensor): The concatenated gt bboxes of all samples
                in the batch, has shape (num_target_total, 4) with the last
                dimension arranged as (cx, cy, w, h) where
                `num_target_total = sum(num_target_list)`.
            num_groups (int): The number of denoising query groups.

        Returns:
            Tensor: The output noisy bboxes, which are embedded by normalized
            (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
            shape (num_noisy_targets, 4) with the last dimension arranged as
            (cx, cy, w, h), where
            `num_noisy_targets = num_target_total * num_groups * 2`.
        """
        assert self.box_noise_scale > 0
        device = gt_bboxes.device

        # expand gt_bboxes as groups
        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)  # xyxy

        # obtain index of negative queries in gt_bboxes_expand
        positive_idx = torch.arange(
            len(gt_bboxes), dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * len(gt_bboxes) * torch.arange(
            num_groups, dtype=torch.long, device=device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(gt_bboxes)

        # determine the sign of each element in the random part of the added
        # noise to be positive or negative randomly. torch.randint_like -> 随机生成整数
        rand_sign = torch.randint_like(
            gt_bboxes_expand, low=0, high=2,
            dtype=torch.float32) * 2.0 - 1.0  # [low, high), 1 or -1, randomly 这一步相当于是制定了偏移方向

        # calculate the random part of the added noise
        rand_part = torch.rand_like(gt_bboxes_expand)  # [0, 1)
        rand_part[negative_idx] += 1.0  # pos: [0, 1); neg: [1, 2)
        rand_part *= rand_sign  # pos: (-1, 1); neg: (-2, -1] U [1, 2)

        # add noise to the bboxes
        bboxes_whwh = bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = gt_bboxes_expand + torch.mul(
            rand_part, bboxes_whwh) * self.box_noise_scale / 2  # xyxy
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = bbox_xyxy_to_cxcywh(noisy_bboxes_expand)

        dn_bbox_query = inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)
        return dn_bbox_query

    def collate_dn_queries(self, input_label_query: Tensor,
                           input_bbox_query: Tensor, batch_idx: Tensor,
                           batch_size: int, num_groups: int) -> Tuple[Tensor]:
        """Collate generated queries to obtain batched dn queries.

        The strategy for query collation is as follow:

        .. code:: text

                    input_queries (num_target_total, query_dim)
            P_A1 P_B1 P_B2 N_A1 N_B1 N_B2 P'A1 P'B1 P'B2 N'A1 N'B1 N'B2
              |________ group1 ________|    |________ group2 ________|
                                         |
                                         V
                      P_A1 Pad0 N_A1 Pad0 P'A1 Pad0 N'A1 Pad0
                      P_B1 P_B2 N_B1 N_B2 P'B1 P'B2 N'B1 N'B2
                       |____ group1 ____| |____ group2 ____|
             batched_queries (batch_size, max_num_target, query_dim)

            where query_dim is 4 for bbox and self.embed_dims for label.
            Notation: _-group 1; '-group 2;
                      A-Sample1(has 1 target); B-sample2(has 2 targets)

        Args:
            input_label_query (Tensor): The generated label queries of all
                targets, has shape (num_target_total, embed_dims) where
                `num_target_total = sum(num_target_list)`.
            input_bbox_query (Tensor): The generated bbox queries of all
                targets, has shape (num_target_total, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_idx (Tensor): The batch index of the corresponding sample
                for each target, has shape (num_target_total).
            batch_size (int): The size of the input batch.
            num_groups (int): The number of denoising query groups.

        Returns:
            tuple[Tensor]: Output batched label and bbox queries.
            - batched_label_query (Tensor): The output batched label queries,
              has shape (batch_size, max_num_target, embed_dims).
            - batched_bbox_query (Tensor): The output batched bbox queries,
              has shape (batch_size, max_num_target, 4) with the last dimension
              arranged as (cx, cy, w, h).
        """
        device = input_label_query.device
        num_target_list = [
            torch.sum(batch_idx == idx) for idx in range(batch_size)
        ]
        max_num_target = max(num_target_list)
        num_denoising_queries = int(max_num_target * 2 * num_groups)

        map_query_index = torch.cat([
            torch.arange(num_target, device=device)
            for num_target in num_target_list
        ])
        map_query_index = torch.cat([
            map_query_index + max_num_target * i for i in range(2 * num_groups)
        ]).long()
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)

        batched_label_query = torch.zeros(
            batch_size, num_denoising_queries, self.embed_dims, device=device)
        batched_bbox_query = torch.zeros(
            batch_size, num_denoising_queries, 4, device=device)

        batched_label_query[mapper] = input_label_query
        batched_bbox_query[mapper] = input_bbox_query
        return batched_label_query, batched_bbox_query

    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.

        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(max_num_target * 2 * i,
                              max_num_target * 2 * (i + 1))
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        return attn_mask
