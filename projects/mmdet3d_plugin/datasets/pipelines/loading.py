import numpy as np
from mmdet.datasets.builder import PIPELINES
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


@PIPELINES.register_module()
class LoadMultiRadarFromFiles:
    """Load radar points.

    Args:
        load_dim (int): Dimension number of the loaded points. Defaults to 18.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 3, 4].
    """

    def __init__(self, to_float32=False, load_dim=18, use_dim=None):
        self.load_dim = load_dim
        if use_dim is None:
            use_dim = [0, 1, 2]
        self.use_dim = use_dim
        self.to_float32 = to_float32

    def __call__(self, results):
        filename = results["radar_filename"]
        radar = []
        for name, radar2lidar_r, radar2lidar_t in zip(
            filename, results["radar2lidar_rs"], results["radar2lidar_ts"]
        ):
            radar_point_cloud = RadarPointCloud.from_file(name)
            points = radar_point_cloud.points
            points = points.transpose()
            points = np.copy(points).reshape(-1, self.load_dim)
            points = points[:, self.use_dim]
            points = points @ radar2lidar_r + radar2lidar_t
            radar.append(points)

        radar = np.concatenate(radar, axis=0)
        if self.to_float32:
            radar = radar.astype(np.float32)
        results["radar_points"] = radar
        return results
