"""Observed-camera loading and occupancy targets for nuScenes."""
import os
import os.path as osp
import random
import numpy as np
from numpy.linalg import inv
import torch
import mmcv
from mmcv.runner import get_dist_info
from mmdet.datasets.builder import PIPELINES
from nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
_NUSCENES_APIS = {}


def _get_nuscenes_api():
    """Create one NuScenes API per process and resolved dataset root."""
    dataroot = os.environ.get(
        'DUPLEXWORLD_DATA_ROOT',
        os.environ.get('Q4OCC_DATA_ROOT', 'data/nuscenes'))
    dataroot = osp.abspath(osp.expanduser(dataroot))
    api = _NUSCENES_APIS.get(dataroot)
    if api is None:
        api = NuScenes(
            version='v1.0-trainval', dataroot=dataroot, verbose=False)
        _NUSCENES_APIS[dataroot] = api
    return api



def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past,
                      cam_intrinsic_past):
    
    R = sensor2global_rotation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T = sensor2global_translation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T -= ego2global_translation_curr @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T) + lidar2ego_translation_curr @ inv(lidar2ego_rotation_curr).T

    lidar2cam_r = inv(R.T)
    lidar2cam_t = T @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2img



def T_fut2cur(current_ego_pose, future_ego_pose):
    
    q_cur = Quaternion(current_ego_pose['rotation'])
    R_cur = q_cur.rotation_matrix
    t_cur = np.array(current_ego_pose['translation'])

    q_fut = Quaternion(future_ego_pose['rotation'])
    R_fut = q_fut.rotation_matrix
    t_fut = np.array(future_ego_pose['translation'])

    #  R_fut2cur  t_fut2cur
    R_cur_inv = R_cur.T  
    R_fut2cur = R_cur_inv @ R_fut
    t_fut2cur = R_cur_inv @ (t_fut - t_cur)

    # 4x4 T_fut2cur
    T_fut2cur = np.eye(4)
    T_fut2cur[:3, :3] = R_fut2cur
    T_fut2cur[:3, 3] = t_fut2cur
    T_fut2cur = T_fut2cur.astype(np.float32)

    return T_fut2cur



def generate_random_occ_index(n, a, b):
    random_occ_list = [0]
    tmp = sorted(random.sample(range(a, b), n))
    for i in tmp:
        random_occ_list.append(i)
    
    return random_occ_list



def _resolve_future_sample_indices(nusc_api, start_sample_idx, future_offsets):
    """Resolve fixed-horizon sample tokens and mark scene-end padding.

    The released loader keeps the last sample token when a scene has no next
    sample.  Keeping that token is useful for fixed-shape collation, but it is
    not a real future target.  Return an explicit validity bit for every
    requested horizon (including the always-valid current frame) so training
    cannot silently supervise those padded rows.
    """
    offsets = tuple(int(value) for value in future_offsets)
    if not offsets or offsets[0] != 0:
        raise ValueError('future_offsets must start with the current frame 0')
    if any(value < 0 for value in offsets):
        raise ValueError('future_offsets must be non-negative')
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError('future_offsets must be strictly increasing')

    sample_indices = [start_sample_idx]
    future_valid = [True]
    requested = set(offsets[1:])
    sample_idx = start_sample_idx
    has_exact_future = True
    for offset in range(1, offsets[-1] + 1):
        if has_exact_future:
            next_sample_idx = nusc_api.get('sample', sample_idx)['next']
            if next_sample_idx:
                sample_idx = next_sample_idx
            else:
                has_exact_future = False
        if offset in requested:
            sample_indices.append(sample_idx)
            future_valid.append(bool(has_exact_future))

    if len(sample_indices) != len(offsets):
        raise RuntimeError('future sample resolution did not cover all offsets')
    return sample_indices, future_valid



@PIPELINES.register_module()
class LoadOccFromFile:

    def __init__(self, occ_root, future_frames=[0], pred_traj=False, ignore_class_names=[]):
        self.occ_root = occ_root
        self.future_frames = future_frames
        self.pred_traj = pred_traj
        self.ignore_class_names = ignore_class_names
        self.occ_class_names = [
            'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk',
            'terrain', 'manmade', 'vegetation', 'free'
        ]

    def __call__(self, results):
        
        if self.future_frames == None:
            # fut_nums = random.randint(3, 6)
            fut_nums = 3  # it depends on your GPU memory
            fut_list = generate_random_occ_index(fut_nums, 1, 11)  # 21
        else:
            fut_list = self.future_frames

        fut_list = list(fut_list)
        occ_file_list = []
        fut2cur_list = []
        semantics_list = []
        mask_lidar_list = []
        mask_camera_list = []
        fut2cur_list.append(np.eye(4).astype(np.float32))

        scene_name, sample_idx = results['scene_name'], results['sample_idx']
        nusc = _get_nuscenes_api()
        sample_indices, future_valid = _resolve_future_sample_indices(
            nusc, sample_idx, fut_list
        )
        occ_file_list.extend(
            osp.join(self.occ_root, scene_name, token, 'labels.npz')
            for token in sample_indices
        )

        current_sample = nusc.get('sample', sample_idx)
        cur_cam_token = current_sample['data']['CAM_FRONT']
        cur_cam_data = nusc.get('sample_data', cur_cam_token)
        cur_ego_pose = nusc.get('ego_pose', cur_cam_data['ego_pose_token'])

        # if use pred trajs
        # if self.pred_traj:
        #     pred_poses = pred_trajs[sample_idx]['trajectory']

        for future_sample_idx in sample_indices[1:]:
            fut_sample = nusc.get('sample', future_sample_idx)
            fut_cam_token = fut_sample['data']['CAM_FRONT']
            fut_cam_data = nusc.get('sample_data', fut_cam_token)
            fut_ego_pose = nusc.get('ego_pose', fut_cam_data['ego_pose_token'])

            fut2cur = T_fut2cur(cur_ego_pose, fut_ego_pose)
            fut2cur_list.append(fut2cur)
        
        # load lidar and camera visible label
        for occ_file in occ_file_list:
            occ_labels = np.load(occ_file)
            mask_lidar = occ_labels['mask_lidar'].astype(np.bool_)  # [200, 200, 16]
            mask_camera = occ_labels['mask_camera'].astype(np.bool_)  # [200, 200, 16]
            mask_lidar_list.append(mask_lidar)
            mask_camera_list.append(mask_camera)

            semantics = occ_labels['semantics']  # [200, 200, 16]
            for class_id in range(len(self.occ_class_names) - 1):
                mask = semantics == class_id
                if mask.sum() == 0:
                    continue
                if self.occ_class_names[class_id] in self.ignore_class_names:
                    semantics[mask] = self.num_classes - 1
            semantics_list.append(semantics)

        results['fut_list'] = fut_list
        results['fut2cur'] = fut2cur_list
        results['future_valid'] = future_valid
        results['mask_lidar'] = mask_lidar_list
        results['mask_camera'] = mask_camera_list
        results['voxel_semantics'] = semantics_list
        return results



@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps:
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 train_interval=[4, 8],
                 test_interval=6,
                 force_offline=False):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode
        self.force_offline = force_offline

        self.train_interval = train_interval
        self.test_interval = test_interval

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['cam_sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['cam_sweeps']['prev'])
                choices = list(range(len(results['cam_sweeps']['prev']))) + \
                    [len(results['cam_sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['cam_sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    extrinsics = np.eye(4)
                    extrinsics[:3, :3] = sweep[sensor]["sensor2ego_rotation"]
                    extrinsics[:3, 3] = np.array(sweep[sensor]["sensor2ego_translation"])
                    results['extrinsics'].append(extrinsics)
                    intrinsics = np.eye(4)
                    intrinsics[:sweep[sensor]['cam_intrinsic'].shape[0], :sweep[sensor]['cam_intrinsic'].shape[1]] = sweep[sensor]['cam_intrinsic']
                    results['intrinsics'].append(intrinsics)

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval % 6 == 0

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    # results['lidar2img'].append(np.copy(results['cam_intrinsic'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
                    
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    # skip loading history frames
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    extrinsics = np.eye(4)
                    extrinsics[:3, :3] = sweep[sensor]["sensor2ego_rotation"]
                    extrinsics[:3, 3] = np.array(sweep[sensor]["sensor2ego_translation"])
                    intrinsics = np.eye(4)
                    intrinsics[:sweep[sensor]['cam_intrinsic'].shape[0], :sweep[sensor]['cam_intrinsic'].shape[1]] = sweep[sensor]['cam_intrinsic']
                    results['intrinsics'].append(intrinsics)

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        if world_size == 1 and self.test_mode and (not self.force_offline):
            return self.load_online(results)
        else:
            return self.load_offline(results)

