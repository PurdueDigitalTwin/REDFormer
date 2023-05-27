# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
# Modified by Zhiqi Li
# Modified by Can Cui, Yunsheng Ma
# ---------------------------------------------

import mmcv
import numpy as np
import os
import json
import re
from collections import OrderedDict
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
from nuscenes.utils.geometry_utils import view_points
from mmdet3d.datasets import NuScenesDataset
import pickle

from tools.data_converter.nuscenes_converter import get_available_scenes, obtain_sensor2top

nus_categories = ('car',
                  'truck',
                  'trailer',
                  'bus',
                  'construction_vehicle',
                  'bicycle',
                  'motorcycle',
                  'pedestrian',
                  'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider',
                  'cycle.without_rider',
                  'pedestrian.moving',
                  'pedestrian.standing',
                  'pedestrian.sitting_lying_down',
                  'vehicle.moving',
                  'vehicle.parked',
                  'vehicle.stopped',
                  'None')


def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscenes dataset.
    Given the raw data, generate its related info file in pkl format.
    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])
    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path, '{}_infos_ext_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path, '{}_infos_ext_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path, '{}_infos_ext_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)

    ## create rain and night data
    if info_prefix == "nuscenes":
        val_path = osp.join(out_path, '{}_infos_ext_val.pkl'.format(info_prefix))
        with open(val_path, 'rb') as f:
            # Load the data from the file using pickle.load
            data = pickle.load(f)

        data_rain = {'infos': [], 'metadata': {'version': 'v1.0-trainval'}}
        data_night = {'infos': [], 'metadata': {'version': 'v1.0-trainval'}}

        for i in range(len(data['infos'])):
            if data['infos'][i]['rain'] == True:
                data_rain['infos'].append(data['infos'][i])
            if data['infos'][i]['time_of_day'] == True:
                data_night['infos'].append(data['infos'][i])

        with open(osp.join(out_path,'nuscenes_infos_ext_rain_val.pkl'), 'wb') as file:
            pickle.dump(data_rain, file)

        with open(osp.join(out_path,'nuscenes_infos_ext_night_val.pkl'), 'wb') as file:
            pickle.dump(data_night, file)



def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be larger than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        # can_bus.extend(pose[key])  # 16 elements
        can_bus.extend(last_pose[key])
    can_bus.extend([0., 0.])
    return np.array(can_bus)


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes: set,
                         val_scenes: set,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.
    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.
    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])  # sample_data
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])  # calibrated_sensor
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])  # ego_pose
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        mmcv.check_file_exist(lidar_path)
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'radars': dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'time_of_day': bool,
            'rain': bool,
            'ego_velo': float,
        }

        scene_description = nusc.get('scene', sample['scene_token'])['description']
        scene_rain = re.search('.*rain', scene_description, re.IGNORECASE) is not None
        scene_night = re.search('.*night', scene_description, re.IGNORECASE) is not None
        info['rain'] = scene_rain
        info['time_of_day'] = scene_night

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 images information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            # Obtain the info with RT matric from general sensor to Top LiDAR
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain 5 radars information per frame
        radar_types = [
            'RADAR_FRONT',
            'RADAR_FRONT_LEFT',
            'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT',
            'RADAR_BACK_RIGHT'
        ]
        for radar in radar_types:
            radar_token = sample['data'][radar]
            radar_path, _, radar_intrinsic = nusc.get_sample_data(radar_token)
            radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, radar)
            radar_info.update(radar_intrinsic=radar_intrinsic)
            info['radars'].update({radar: radar_info})

        # # obtain sweeps for a single key-frame
        # sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        # sweeps = []
        # while len(sweeps) < max_sweeps:
        #     if not sd_rec['prev'] == '':
        #         sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
        #         sweeps.append(sweep)
        #         sd_rec = nusc.get('sample_data', sd_rec['prev'])
        #     else:
        #         break
        # info['sweeps'] = sweeps

        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]
            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)

            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)

            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


if __name__ == '__main__':
    create_nuscenes_infos(
        root_path="data/nuscenes/full",
        out_path="data/nuscenes/full",
        can_bus_root_path="data/nuscenes/full",
        info_prefix="nuscenes",
        version="v1.0-trainval",
        max_sweeps=10)

