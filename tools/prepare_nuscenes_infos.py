#!/usr/bin/env python3
"""Add camera sweep/pose and occupancy references to legacy nuScenes info PKLs.

Input: the standard MMDetection3D train/val info dictionaries (infos, metadata).
Output: nuscenes_infos_{train,val}_sweep_occ.pkl. No dataset files are copied.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pickle

CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


def camera_info(nusc, record, data_root):
    import numpy as np
    from pyquaternion import Quaternion
    pose=nusc.get('ego_pose',record['ego_pose_token'])
    sensor=nusc.get('calibrated_sensor',record['calibrated_sensor_token'])
    s2e_r=Quaternion(sensor['rotation']).rotation_matrix
    e2g_r=Quaternion(pose['rotation']).rotation_matrix
    s2e_t=np.array(sensor['translation'])
    e2g_t=np.array(pose['translation'])
    return dict(data_path=str(data_root/record['filename']),
                sensor2global_rotation=s2e_r.T@e2g_r.T,
                sensor2global_translation=s2e_t@e2g_r.T+e2g_t,
                sensor2ego_rotation=s2e_r,sensor2ego_translation=sensor['translation'],
                cam_intrinsic=np.array(sensor['camera_intrinsic']),timestamp=record['timestamp'])


def augment(nusc, payload, data_root, occ_root):
    from tqdm import tqdm
    if not isinstance(payload,dict) or 'infos' not in payload:
        raise ValueError('Expected a legacy MMDetection3D info dictionary containing infos/metadata.')
    for info in tqdm(payload['infos'],desc='Camera sweeps'):
        sample=nusc.get('sample',info['token'])
        scene=nusc.get('scene',sample['scene_token'])
        info.update(scene_name=scene['name'],prev=sample['prev'],next=sample['next'],
                    occ_file=str(occ_root/scene['name']/info['token']/'labels.npz'))
        current={cam:nusc.get('sample_data',sample['data'][cam]) for cam in CAMERAS}
        for cam in CAMERAS:
            info['cams'][cam].update(camera_info(nusc,current[cam],data_root))
        sweeps=[]
        if sample['prev']:
            for _ in range(5):
                sweep={}
                for cam in CAMERAS:
                    if not current[cam]['prev']:
                        if not sweeps:
                            raise ValueError(f"Missing first camera sweep for sample {info['token']} ({cam})")
                        sweep=sweeps[-1]
                        break
                    record=nusc.get('sample_data',current[cam]['prev'])
                    current[cam]=record
                    sweep[cam]=camera_info(nusc,record,data_root)
                sweeps.append(sweep)
        if 'sweeps' in info:
            info['lidar_sweeps']=info.pop('sweeps')
        elif 'lidar_sweeps' not in info:
            raise ValueError('Missing sweeps field in standard input info PKL.')
        info['cam_sweeps']=sweeps
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--occ-root')
    p.add_argument('--base-info-dir',required=True,help='directory with nuscenes_infos_train.pkl and nuscenes_infos_val.pkl')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    from nuscenes import NuScenes
    data=Path(args.data_root).expanduser().resolve()
    occ=Path(args.occ_root).expanduser().resolve() if args.occ_root else data/'gts'
    output=Path(args.output_dir).expanduser().resolve()
    base=Path(args.base_info_dir).expanduser().resolve()
    pairs=[(base/f'nuscenes_infos_{s}.pkl',output/f'nuscenes_infos_{s}_sweep_occ.pkl') for s in ('train','val')]
    for src,dst in pairs:
        if not src.is_file(): raise FileNotFoundError(src)
        if dst.exists() and not args.overwrite: raise FileExistsError(f'{dst} exists. Use --overwrite to replace it.')
    nusc=NuScenes('v1.0-trainval',str(data),verbose=True)
    output.mkdir(parents=True,exist_ok=True)
    for src,dst in pairs:
        with src.open('rb') as f: payload=pickle.load(f)
        payload=augment(nusc,payload,data,occ)
        temporary=dst.with_name(dst.name+f'.tmp.{os.getpid()}')
        try:
            with temporary.open('wb') as f:pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary,dst)
        finally:
            if temporary.exists():temporary.unlink()
        print(f'Saved {dst}',flush=True)


if __name__=='__main__':
    main()
