import os
import mmcv
import numpy as np
import os.path as osp
from pathlib import Path
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix


def _portable_nuscenes_path(path, data_root):
    """Rebase paths serialized by another NuScenes server installation."""
    raw = Path(os.path.expanduser(str(path)))
    root = Path(os.path.expanduser(str(data_root))).resolve()
    parts = tuple(part for part in raw.parts if part not in ('', '.'))
    for marker in (('data', 'nuscenes'), ('samples',), ('sweeps',), ('maps',)):
        width = len(marker)
        for index in range(max(0, len(parts) - width + 1)):
            if parts[index:index + width] == marker:
                suffix_index = index + width if width == 2 else index
                return str(root.joinpath(*parts[suffix_index:]))
    return str(raw if raw.is_absolute() else root / raw)


def _rebase_sweeps(sweeps, data_root):
    for sweep in sweeps:
        if isinstance(sweep, dict) and 'data_path' in sweep:
            sweep['data_path'] = _portable_nuscenes_path(
                sweep['data_path'], data_root)
        elif isinstance(sweep, dict):
            for sensor in sweep.values():
                if isinstance(sensor, dict) and 'data_path' in sensor:
                    sensor['data_path'] = _portable_nuscenes_path(
                        sensor['data_path'], data_root)
    return sweeps


@DATASETS.register_module()
class NuScenesOccDataset(NuScenesDataset):    
    def __init__(self, future_frames, *args, **kwargs):
        super().__init__(filter_empty_gt=False, *args, **kwargs)
        self.data_infos = self.load_annotations(self.ann_file)
        self.future_frames = future_frames
        if self.future_frames != None:
            print('future_frames: ', self.future_frames)
        else:
            print('Using random future frames!')
    
    def collect_cam_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def collect_lidar_sweeps(self, index, into_past=20, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['lidar_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        last_timestamp = self.data_infos[index]['timestamp']
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['lidar_sweeps'][::-1]
            if curr_sweeps[0]['timestamp'] == last_timestamp:
                curr_sweeps = curr_sweeps[1:]
            all_sweeps_next.extend(curr_sweeps)
            curr_index = curr_index + 1
            last_timestamp = all_sweeps_next[-1]['timestamp']

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix
        ego2lidar = transform_matrix(
            lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)

        input_dict = dict(
            sample_idx=info['token'],
            scene_name=info['scene_name'],
            timestamp=info['timestamp'] / 1e6,
            ego2lidar=ego2lidar,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        if self.modality['use_lidar']:
            lidar_sweeps_prev, lidar_sweeps_next = self.collect_lidar_sweeps(index)
            lidar_sweeps_prev = _rebase_sweeps(
                lidar_sweeps_prev, self.data_root)
            lidar_sweeps_next = _rebase_sweeps(
                lidar_sweeps_next, self.data_root)
            input_dict.update(dict(
                pts_filename=_portable_nuscenes_path(
                    info['lidar_path'], self.data_root),
                lidar_sweeps={'prev': lidar_sweeps_prev, 'next': lidar_sweeps_next},
            ))

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            lidar2cam_rts = []
            intrinsics = []
            extrinsics = []

            for _, cam_info in info['cams'].items():
                img_paths.append(_portable_nuscenes_path(
                    cam_info['data_path'], self.data_root))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt)
                intrinsics.append(viewpad)

                c2e = np.eye(4)
                c2e[:3, :3] = cam_info["sensor2ego_rotation"]
                c2e[:3, 3] = np.array(cam_info["sensor2ego_translation"])
                extrinsics.append(c2e)

            cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(index)
            cam_sweeps_prev = _rebase_sweeps(
                cam_sweeps_prev, self.data_root)
            cam_sweeps_next = _rebase_sweeps(
                cam_sweeps_next, self.data_root)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                lidar2cam=lidar2cam_rts,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                cam_sweeps={'prev': cam_sweeps_prev, 'next': cam_sweeps_next},
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
