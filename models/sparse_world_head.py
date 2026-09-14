import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn, Voxelization
from mmcv.cnn import xavier_init
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from .bbox.utils import decode_points
# from .utils import calc_dcd
# from .metrics import cd
# cham_loss = cd()
import math
# import time


@HEADS.register_module()
class SparseWorldHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 future_frames,
                 transformer=None,
                 pc_range=[],
                 empty_label=17,
                 voxel_size=[],
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),  # dict(type='L1Loss'), loss_pts='dcd'
                 init_cfg=None,
                 use_can_bus=False,
                 q4occ_screening_loss=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.future_frames = future_frames
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.loss_cls = build_loss(loss_cls)
        # 'dcd': density-aware chamfer distance
        # if loss_pts == 'dcd':
        #     self.loss_pts = calc_dcd
        # else:
        # vanilla chamfer distance via pair-wise SmoothL1Loss
        self.loss_pts = build_loss(loss_pts)
        self.q4occ_screening_loss_enabled = q4occ_screening_loss is not None
        self.q4occ_screening_loss = self._validate_screening_loss(
            q4occ_screening_loss
        )
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.voxel_generator = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=10, 
            max_voxels=self.num_query * self.num_refines[-1],
            deterministic=False
        )

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.voxel_shape = tuple(int(value) for value in voxel_num.tolist())
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        self._init_layers()

        # use_can_bus for future prediction
        self.use_can_bus = use_can_bus
        if self.use_can_bus:
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(18, self.embed_dims * 2),
                nn.LayerNorm(self.embed_dims * 2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.embed_dims * 2, self.in_channels),
                nn.LayerNorm(self.in_channels),
                nn.LeakyReLU(inplace=True),
            )
            xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    def _validate_screening_loss(self, raw):
        """Validate opt-in A2/A3 losses without changing released defaults.

        A ``None`` default keeps every existing SparseWorld, Q4Occ, K512,
        dual-memory, and DCM-JR configuration on the released loss path
        byte-for-byte.  Any screening mapping opts into mandatory scene-end
        an explicit scene-end target policy and may enable exactly one of:

        * horizon_weights: A2 forecast-aligned relative horizon weights;
        * soft_voxel_weight: A3 final-stage binary soft-IoU auxiliary.
        """

        options = dict(raw or {})
        allowed = {
            'horizon_weights',
            'soft_voxel_weight',
            'soft_voxel_eps',
            'require_future_valid',
            'future_padding_policy',
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(
                'unknown q4occ_screening_loss keys: ' + ', '.join(unknown)
            )

        legacy_requirement = options.get('require_future_valid')
        if legacy_requirement is not None and type(legacy_requirement) is not bool:
            raise TypeError('require_future_valid must be bool')
        padding_policy = options.get('future_padding_policy')
        if padding_policy is None:
            # Preserve the v1 screening behavior for old mappings while making
            # the released padded-target policy explicit in new fresh arms.
            padding_policy = (
                'mask' if legacy_requirement is None or legacy_requirement
                else 'retain'
            )
        padding_policy = str(padding_policy).lower()
        if padding_policy not in {'retain', 'mask'}:
            raise ValueError("future_padding_policy must be 'retain' or 'mask'")
        require_future_valid = padding_policy == 'mask'
        if (
            legacy_requirement is not None
            and legacy_requirement != require_future_valid
        ):
            raise ValueError(
                'require_future_valid conflicts with future_padding_policy'
            )

        horizon_weights = options.get('horizon_weights')
        if horizon_weights is not None:
            if not isinstance(horizon_weights, (list, tuple)):
                raise TypeError('horizon_weights must be a list or tuple')
            horizon_weights = tuple(float(value) for value in horizon_weights)
            if len(horizon_weights) != len(self.future_frames):
                raise ValueError(
                    'horizon_weights must match future_frames: '
                    f'{len(horizon_weights)} != {len(self.future_frames)}'
                )
            if any(not math.isfinite(value) or value <= 0.0
                   for value in horizon_weights):
                raise ValueError('horizon_weights must be finite and positive')

        soft_voxel_weight = float(options.get('soft_voxel_weight', 0.0))
        if (not math.isfinite(soft_voxel_weight)
                or soft_voxel_weight < 0.0
                or soft_voxel_weight > 1.0):
            raise ValueError('soft_voxel_weight must be finite and in [0, 1]')
        soft_voxel_eps = float(options.get('soft_voxel_eps', 1.0e-6))
        if (not math.isfinite(soft_voxel_eps)
                or soft_voxel_eps <= 0.0
                or soft_voxel_eps >= 1.0e-2):
            raise ValueError('soft_voxel_eps must be finite and in (0, 1e-2)')
        if horizon_weights is not None and soft_voxel_weight > 0.0:
            raise ValueError(
                'A2 horizon weighting and A3 soft voxel loss must remain '
                'independent screening arms'
            )
        return dict(
            horizon_weights=horizon_weights,
            soft_voxel_weight=soft_voxel_weight,
            soft_voxel_eps=soft_voxel_eps,
            require_future_valid=require_future_valid,
            future_padding_policy=padding_policy,
        )

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, img_metas, fut2cur, fut_list):
        
        B, Q, = mlvl_feats[0].shape[0], self.num_query
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)
        # query_feat = init_points.new_empty(B, Q, self.embed_dims).uniform_(0, 1)

        cls_scores, refine_pts = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            img_metas=img_metas,
            fut2cur=fut2cur,
            fut_list=fut_list,
        )

        FT = len(fut2cur)
        init_points = init_points.repeat(FT, 1, 1, 1)
        return dict(init_points=init_points,
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts)

    def get_dis_weight(self, pts):
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = (self.pc_range[:3] + self.pc_range[3:]) / 2
        dist = (pts - centers[None, ...])[..., :2]
        dist = torch.norm(dist, dim=-1)
        return dist / max_dist + 1
    
    def discretize(self, pts, clip=True, decode=False):
        loc = torch.floor((pts - self.pc_range[:3]) / self.voxel_size)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)

        return loc.long() if not decode else \
            (loc + 0.5) * self.voxel_size + self.pc_range[:3]

    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_masks, gt_labels):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()  # [num_gt_pts]
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()  # [num_pred_pts]
        gt_paired_pts = refine_pts[gt_paired_idx]  # [num_gt_pts]
        pred_paired_pts = gt_points[pred_paired_idx]  # [num_pred_pts]

        # cls assignment
        refine_pts_labels = gt_labels[pred_paired_idx]  # [num_pred_pts]
        cls_weights = self.train_cfg.get('cls_weights', [1] * self.num_classes)
        cls_weights = refine_pts.new_tensor(cls_weights)  # [17]
        label_weights = cls_weights * \
            self.get_dis_weight(pred_paired_pts)[..., None]  # [num_pred_pts, 17]

        # gt side assignment
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 5)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr) & gt_masks  # [num_gt_pts]
        gt_pts_weights[mask] = empty_weights  # [num_gt_pts]

        rare_classes = self.train_cfg.get('rare_classes', [0, 2, 5, 8])
        # others, bicycle, construction_vehicle, traffic_cone
        rare_weights = self.train_cfg.get('rare_weights', 10)
        for cls_idx in rare_classes:
            mask = (gt_labels == cls_idx) & gt_masks
            gt_pts_weights[mask] = gt_pts_weights[mask].clamp(min=rare_weights)

        return (refine_pts_labels, gt_paired_idx, pred_paired_idx, label_weights, 
                gt_pts_weights)
    
    def get_targets(self):
        # To instantiate the abstract method
        pass

    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list,
                    row_weights=None):
        # start_time_loss = time.perf_counter()
        num_imgs = cls_scores.size(0) # B
        if row_weights is not None:
            row_weights = tuple(float(value) for value in row_weights)
            if len(row_weights) != num_imgs:
                raise ValueError(
                    f'row_weights has {len(row_weights)} values for '
                    f'{num_imgs} prediction rows'
                )
            if any(not math.isfinite(value) or value < 0.0
                   for value in row_weights):
                raise ValueError('row_weights must be finite and non-negative')
            if not any(value > 0.0 for value in row_weights):
                raise ValueError('row_weights must contain a positive value')
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        # ######### prepare for dcd_loss ########
        # start_time_loss = time.perf_counter()
        # pred_pts = torch.cat(refine_pts_list)
        # gt_pts = torch.cat(gt_points_list)
        # dist1, dist2, idx1, idx2 = cham_loss(gt_pts.unsqueeze(0), pred_pts.unsqueeze(0))
        # elapsed_loss = time.perf_counter() - start_time_loss
        # print(f'runtime of calculating cham_loss: {elapsed_loss} s')
        # dist1 = torch.sqrt(dist1).squeeze(0)
        # dist2 = torch.sqrt(dist2).squeeze(0)
        # idx1 = idx1.squeeze(0)  # [num_gt_pts]
        # idx2 = idx2.squeeze(0)  # [num_pred_pts]

        # ## cls assignment
        # gt_labels = torch.cat(gt_labels_list)
        # labels = gt_labels[idx2]
        # cls_weights = self.train_cfg.get('cls_weights', [1] * self.num_classes)
        # cls_weights = refine_pts.new_tensor(cls_weights)  # [17]
        # pred_paired_pts = gt_pts[idx2]
        # label_weights = cls_weights * \
        #     self.get_dis_weight(pred_paired_pts)[..., None]  # [num_pred_pts, 17]

        # ## gt side assignment
        # empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        # empty_weights = self.train_cfg.get('empty_weights', 5)
        # gt_paired_pts = pred_pts[idx1]
        # gt_masks = torch.cat(gt_masks_list)

        # gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        # ## dist = torch.norm(gt_pts - gt_paired_pts, dim=-1)  # torch.sqrt(dist1)
        # mask = (dist1 > empty_dist_thr) & gt_masks  # [num_gt_pts]
        # gt_pts_weights[mask] = empty_weights  # [num_gt_pts]

        # rare_classes = self.train_cfg.get('rare_classes', [0, 2, 5, 8])
        # ## others, bicycle, construction_vehicle, traffic_cone
        # rare_weights = self.train_cfg.get('rare_weights', 10)
        # for cls_idx in rare_classes:
        #     mask = (gt_labels == cls_idx) & gt_masks
        #     gt_pts_weights[mask] = gt_pts_weights[mask].clamp(min=rare_weights)
        # ######### prepare for dcd_loss ########
        
        # start_time_knn = time.perf_counter()
        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
             self._get_target_single, refine_pts_list, gt_points_list, 
             gt_masks_list, gt_labels_list)
        # elapsed_knn = time.perf_counter() - start_time_knn
        # print(f'runtime of calculating loss_knn: {elapsed_knn} s')
        
        gt_paired_pts, pred_paired_pts= [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # Apply A2 weights before concatenation.  Each denominator is the
        # corresponding weighted element count, so [1,1,...] reproduces the
        # released normalization while [0.5,1,1,1] changes relative horizon
        # emphasis without changing the overall loss scale.
        if row_weights is not None:
            cls_weights = [
                weight * row_weights[index]
                for index, weight in enumerate(cls_weights)
            ]
            gt_pts_weights = [
                weight * row_weights[index]
                for index, weight in enumerate(gt_pts_weights)
            ]
            pred_pts_weights = [
                refine_pts_list[index].new_full(
                    (refine_pts_list[index].shape[0], 1),
                    row_weights[index],
                )
                for index in range(num_imgs)
            ]
            cls_avg_factor = sum(
                row_weights[index] * cls_scores_list[index].shape[0]
                for index in range(num_imgs)
            )
            gt_avg_factor = sum(
                row_weights[index] * gt_points_list[index].shape[0]
                for index in range(num_imgs)
            )
            pred_avg_factor = sum(
                row_weights[index] * refine_pts_list[index].shape[0]
                for index in range(num_imgs)
            )
        else:
            pred_pts_weights = None
            cls_avg_factor = sum(value.shape[0] for value in cls_scores_list)
            gt_avg_factor = sum(value.shape[0] for value in gt_points_list)
            pred_avg_factor = sum(value.shape[0] for value in refine_pts_list)

        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)
        if pred_pts_weights is not None:
            pred_pts_weights = torch.cat(pred_pts_weights)

        # calculate loss cls
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights,  # cls_weights, label_weights
                                 avg_factor=cls_avg_factor)
        
        # calculate loss pts
        # start_time_pts = time.perf_counter()
        loss_pts = pred_pts.new_tensor(0)
        # loss_pts += (gt_pts_weights * dist1).mean() + dist2.mean()
        
        # x = pred_pts.unsqueeze(0)
        # gt = gt_pts.unsqueeze(0)
        # start_time_dcd = time.perf_counter()
        # loss_dcd = calc_dcd(x, gt, alpha=40, n_lambda=0.2, non_reg=True).squeeze(0)
        # print(f'loss_dcd: {loss_dcd}')
        # elapsed_dcd = time.perf_counter() - start_time_dcd
        # print(f'runtime of calculating loss_dcd: {elapsed_dcd} s')

        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_avg_factor)
        if pred_pts_weights is None:
            loss_pts += self.loss_pts(
                pred_pts,
                pred_paired_pts,
                avg_factor=pred_avg_factor)
        else:
            loss_pts += self.loss_pts(
                pred_pts,
                pred_paired_pts,
                weight=pred_pts_weights,
                avg_factor=pred_avg_factor)

        # elapsed_loss = time.perf_counter() - start_time_loss
        # print(f'runtime of calculating loss: {elapsed_loss} s')

        return loss_cls, loss_pts

    def _screening_row_weights(self, prediction_rows, future_valid):
        if not self.q4occ_screening_loss_enabled:
            return None
        if prediction_rows != len(self.future_frames):
            raise ValueError(
                'screening loss expects one scene in horizon-major '
                f'order ({len(self.future_frames)} rows), got {prediction_rows}; '
                'use Q4OccPhase1PerformanceBatchSafeHead for local batch > 1'
            )
        horizon_weights = self.q4occ_screening_loss['horizon_weights']
        if horizon_weights is None:
            horizon_weights = (1.0,) * prediction_rows
        padding_policy = self.q4occ_screening_loss.get(
            'future_padding_policy',
            'mask' if self.q4occ_screening_loss.get(
                'require_future_valid', True
            ) else 'retain',
        )
        if padding_policy == 'retain':
            effective_sum = sum(horizon_weights)
            scale = float(prediction_rows) / effective_sum
            return tuple(weight * scale for weight in horizon_weights)

        if future_valid is None:
            raise ValueError(
                'future_valid is required whenever q4occ_screening_loss is set'
            )
        if torch.is_tensor(future_valid):
            if future_valid.ndim != 1 or future_valid.shape[0] != prediction_rows:
                raise ValueError(
                    'future_valid must have one value per horizon row, got '
                    f'{tuple(future_valid.shape)}'
                )
            if future_valid.dtype != torch.bool:
                raise TypeError('future_valid tensor must have bool dtype')
            validity = tuple(
                bool(value)
                for value in future_valid.detach().cpu().tolist()
            )
        elif isinstance(future_valid, (list, tuple)):
            if len(future_valid) != prediction_rows:
                raise ValueError(
                    'future_valid must have one value per horizon row, got '
                    f'{len(future_valid)}'
                )
            if not all(type(value) is bool for value in future_valid):
                raise TypeError('future_valid values must be bool')
            validity = tuple(future_valid)
        else:
            raise TypeError('future_valid must be a bool tensor or sequence')
        if not validity[0]:
            raise ValueError('the current-frame future_valid entry must be true')

        effective = tuple(
            weight if valid else 0.0
            for weight, valid in zip(horizon_weights, validity)
        )
        valid_count = sum(validity)
        effective_sum = sum(effective)
        if valid_count < 1 or effective_sum <= 0.0:
            raise ValueError('future_valid must contain at least one valid row')
        scale = float(valid_count) / effective_sum
        return tuple(weight * scale for weight in effective)

    def _soft_splat_binary_occupancy(self, cls_scores, refine_pts):
        """Trilinear-splat one horizon row with bounded voxel averaging.

        This is deliberately row-streamed by the caller: no ``B x FT`` dense
        volume exists at once.  Integer neighbor selection is piecewise
        constant, while the trilinear source weights preserve coordinate
        gradients inside a voxel.  Dividing ``score_sum`` by
        ``max(mass_sum, 1)`` behaves like a soft version of the released
        within-voxel score average without letting a many-point voxel exceed
        probability one.  For an isolated point ``mass_sum < 1``, so its
        occupancy remains ``probability * trilinear_weight`` and retains the
        intended coordinate gradient.
        """

        if cls_scores.ndim != 3 or refine_pts.ndim != 3:
            raise ValueError(
                'soft voxel loss expects [query,point,channel] predictions'
            )
        if (tuple(cls_scores.shape[:-1]) != tuple(refine_pts.shape[:-1])
                or cls_scores.shape[-1] != self.num_classes
                or refine_pts.shape[-1] != 3):
            raise ValueError('classification and point prediction shapes differ')

        cls_scores = cls_scores.float()
        refine_pts = refine_pts.float()
        refine_pts = decode_points(refine_pts, self.pc_range)
        centers = refine_pts.mean(dim=1, keepdim=True)
        center_distance = torch.norm(refine_pts - centers, dim=-1)
        ctr_dist_thr = float(self.test_cfg.get('ctr_dist_thr', 3.0))
        if not math.isfinite(ctr_dist_thr) or ctr_dist_thr <= 0.0:
            raise ValueError('test_cfg.ctr_dist_thr must be finite and positive')
        # Match the released inference support without introducing a gradient
        # through the discrete keep/drop decision at the radius boundary.
        within_query_support = (
            torch.isfinite(center_distance.detach())
            & (center_distance.detach() < ctr_dist_thr)
        ).reshape(-1)

        occupancy_probability = (
            cls_scores.sigmoid().amax(dim=-1).reshape(-1)
        )
        occupancy_probability = torch.nan_to_num(
            occupancy_probability, nan=0.0, posinf=1.0, neginf=0.0
        )
        refine_pts = refine_pts.reshape(-1, 3)
        continuous = (
            (refine_pts - self.pc_range[:3]) / self.voxel_size - 0.5
        )
        finite = (
            torch.isfinite(continuous).all(dim=-1)
            & within_query_support
        )
        continuous = torch.nan_to_num(
            continuous, nan=0.0, posinf=0.0, neginf=0.0
        )
        base = torch.floor(continuous)
        fraction = (continuous - base).clamp(0.0, 1.0)
        base = base.long()

        flat_indices, score_contributions, mass_contributions = [], [], []
        voxel_x, voxel_y, voxel_z = self.voxel_shape
        for offset_x in (0, 1):
            weight_x = fraction[:, 0] if offset_x else 1.0 - fraction[:, 0]
            for offset_y in (0, 1):
                weight_y = (
                    fraction[:, 1] if offset_y else 1.0 - fraction[:, 1]
                )
                for offset_z in (0, 1):
                    weight_z = (
                        fraction[:, 2]
                        if offset_z else 1.0 - fraction[:, 2]
                    )
                    location = base + base.new_tensor(
                        [offset_x, offset_y, offset_z]
                    )
                    valid = (
                        finite
                        & (location[:, 0] >= 0)
                        & (location[:, 0] < voxel_x)
                        & (location[:, 1] >= 0)
                        & (location[:, 1] < voxel_y)
                        & (location[:, 2] >= 0)
                        & (location[:, 2] < voxel_z)
                    )
                    safe = location.clone()
                    safe[:, 0].clamp_(0, voxel_x - 1)
                    safe[:, 1].clamp_(0, voxel_y - 1)
                    safe[:, 2].clamp_(0, voxel_z - 1)
                    flat_indices.append(
                        safe[:, 0] * (voxel_y * voxel_z)
                        + safe[:, 1] * voxel_z
                        + safe[:, 2]
                    )
                    mass = weight_x * weight_y * weight_z
                    mass = mass * valid.to(mass.dtype)
                    mass_contributions.append(mass)
                    score_contributions.append(
                        occupancy_probability * mass
                    )

        flat_indices = torch.cat(flat_indices)
        score_contributions = torch.cat(score_contributions)
        mass_contributions = torch.cat(mass_contributions)
        voxel_count = voxel_x * voxel_y * voxel_z
        score_sum = score_contributions.new_zeros(voxel_count).scatter_add(
            0, flat_indices, score_contributions
        )
        mass_sum = mass_contributions.new_zeros(voxel_count).scatter_add(
            0, flat_indices, mass_contributions
        )
        occupancy = score_sum / mass_sum.clamp_min(1.0)
        occupancy = torch.nan_to_num(
            occupancy, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        return occupancy.reshape(voxel_x, voxel_y, voxel_z)

    def _soft_binary_voxel_iou_loss(
        self,
        cls_scores,
        refine_pts,
        voxel_semantics,
        mask_camera,
        row_weights,
    ):
        rows = int(cls_scores.shape[0])
        expected_grid = self.voxel_shape
        if tuple(voxel_semantics.shape) != (rows, *expected_grid):
            raise ValueError(
                'voxel_semantics shape does not match final predictions/grid: '
                f'{tuple(voxel_semantics.shape)} != {(rows, *expected_grid)}'
            )
        if tuple(mask_camera.shape) != tuple(voxel_semantics.shape):
            raise ValueError('mask_camera must match voxel_semantics')

        eps = self.q4occ_screening_loss['soft_voxel_eps']
        row_losses = []
        valid_weights = []
        for row in range(rows):
            if row_weights[row] <= 0.0:
                continue
            predicted = self._soft_splat_binary_occupancy(
                cls_scores[row], refine_pts[row]
            )
            visible = mask_camera[row].to(dtype=torch.bool)
            target = (
                voxel_semantics[row].long() != self.empty_label
            ).to(predicted.dtype)
            visible_float = visible.to(predicted.dtype)
            intersection = (predicted * target * visible_float).sum()
            union = (
                (predicted + target - predicted * target) * visible_float
            ).sum()
            row_losses.append(1.0 - (intersection + eps) / (union + eps))
            valid_weights.append(row_weights[row])
        if not row_losses:
            raise ValueError('soft voxel loss requires at least one valid row')
        loss_weights = row_losses[0].new_tensor(valid_weights)
        return (
            torch.stack(row_losses) * loss_weights
        ).sum() / loss_weights.sum()
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(
        self,
        voxel_semantics,
        mask_camera,
        preds_dicts,
        future_valid=None,
    ):
        # voxelsemantics [B, X200, Y200, Z16] unocuupied=17
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores']  # 6, [B, Q, _, 17], B=8*FT
        all_refine_pts = preds_dicts['all_refine_pts']  # 6, [B, Q, _, 3]

        num_dec_layers = len(all_cls_scores)  # 6 layers
        row_weights = self._screening_row_weights(
            all_cls_scores[0].shape[0], future_valid
        )
        gt_points_list, gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics, mask_camera)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        if row_weights is None:
            # Preserve the released default call graph exactly.
            losses_cls, losses_pts = multi_apply(
                self.loss_single, all_cls_scores, all_refine_pts,
                all_gt_points_list, all_gt_masks_list, all_gt_labels_list)
        else:
            weighted_losses = [
                self.loss_single(
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list,
                    row_weights=row_weights,
                )
                for cls_scores, refine_pts in zip(
                    all_cls_scores, all_refine_pts
                )
            ]
            losses_cls = [value[0] for value in weighted_losses]
            losses_pts = [value[1] for value in weighted_losses]

        loss_dict = dict()
        # loss of init_points
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, gt_points_list, 
                gt_masks_list, gt_labels_list,
                row_weights=row_weights)
            loss_dict['init_loss_pts'] = init_loss_pts

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls[:-1], losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        soft_voxel_weight = self.q4occ_screening_loss['soft_voxel_weight']
        if soft_voxel_weight > 0.0:
            loss_dict['loss_voxel_binary_soft_iou'] = (
                soft_voxel_weight
                * self._soft_binary_voxel_iou_loss(
                    all_cls_scores[-1],
                    all_refine_pts[-1],
                    voxel_semantics,
                    mask_camera,
                    row_weights,
                )
            )
        return loss_dict
    
    def get_occ(self, pred_dicts, img_metas, rescale=False):
        all_cls_scores = pred_dicts['all_cls_scores']
        all_refine_pts = pred_dicts['all_refine_pts']
        cls_scores = all_cls_scores[-1].sigmoid()  # torch.Size([B, 600, 128, 17])
        refine_pts = all_refine_pts[-1]  # torch.Size([B, 600, 128, 3])

        batch_size = refine_pts.shape[0]  # B = len([0, 2, 4, 6])
        ctr_dist_thr = self.test_cfg.get('ctr_dist_thr', 3.)  # 3.0
        score_thr = self.test_cfg.get('score_thr', 0.)  # 0.5

        result_list = []
        for i in range(batch_size):
            refine_pt, cls_score = refine_pts[i], cls_scores[i]
            refine_pt = decode_points(refine_pt, self.pc_range)

            # filter weak points by distance and score
            centers = refine_pt.mean(dim=1, keepdim=True)  # [600, 1, 3]
            ctr_dists = torch.norm(refine_pt - centers, dim=-1)  # [Q, P], [600, 128]
            mask_dist = ctr_dists < ctr_dist_thr
            mask_score = (cls_score > score_thr).any(dim=-1)
            mask = mask_dist & mask_score
            refine_pt = refine_pt[mask]
            cls_score = cls_score[mask]

            pts = torch.cat([refine_pt, cls_score], dim=-1)
            pts_infos, voxels, num_pts = self.voxel_generator(pts)
            voxels = torch.flip(voxels, [1]).long()
            pts, scores = pts_infos[..., :3], pts_infos[..., 3:]
            scores = scores.sum(dim=1) / num_pts[..., None]

            if self.test_cfg.get('padding', True):
                occ = scores.new_zeros((self.voxel_num[0], self.voxel_num[1], 
                                        self.voxel_num[2], self.num_classes))
                occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = scores
                occ = occ.permute(3, 0, 1, 2).unsqueeze(0)
                # padding
                dilated_occ = F.max_pool3d(occ, 3, stride=1, padding=1)
                eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)
                # repalce with original occ prediction
                original_mask = (occ > score_thr).any(dim=1, keepdim=True)
                original_mask = original_mask.expand_as(eroded_occ)
                eroded_occ[original_mask] = occ[original_mask]
                # sparse dense occ
                eroded_occ = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
                voxels = torch.nonzero((eroded_occ > score_thr).any(dim=-1))
                scores = eroded_occ[voxels[:, 0], voxels[:, 1], voxels[:, 2], :]

            labels = scores.argmax(dim=-1)
            result_list.append(dict(
                sem_pred=labels.detach().cpu().numpy(),
                occ_loc=voxels.detach().cpu().numpy()))

        return result_list
    
    def get_sparse_voxels(self, voxel_semantics, mask_camera):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        coors = torch.stack([xx, yy, zz], dim=-1) # actual space

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.empty_label
            gt_points.append(coors[mask])
            gt_masks.append(mask_camera[i][mask]) # camera mask and not empty
            gt_labels.append(voxel_semantics[i][mask])
        
        return gt_points, gt_masks, gt_labels
