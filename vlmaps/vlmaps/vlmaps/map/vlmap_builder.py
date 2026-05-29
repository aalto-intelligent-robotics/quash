import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Set, Optional

from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig
import torch
import gdown
import open3d as o3d
from collections import defaultdict

from vlmaps.utils.mapping_utils import (
    load_3d_map,
    save_3d_map,
    cvt_pose_vec2tf,
    load_depth_npy,
    load_semantic_npy,
    depth2pc,
    transform_pc,
    base_pos2grid_id_3d,
    project_point,
    get_sim_cam_mat,
)
from embeddings.pixelEmbeddingCreator import PixelEmbeddingCreator


def visualize_pc(pc: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])


class VLMapBuilder:
    def __init__(
        self,
        data_dir: Path,
        map_config: DictConfig,
        pose_path: Path,
        rgb_paths: List[Path],
        depth_paths: List[Path],
        base2cam_tf: np.ndarray,
        base_transform: np.ndarray,
        semantic_paths: Path,
        region_paths: Path,
        force: bool,
        store_all: bool
    ):
        self.data_dir = data_dir
        self.pose_path = pose_path
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.map_config = map_config
        self.base2cam_tf = base2cam_tf
        self.base_transform = base_transform
        self.semantic_paths = semantic_paths
        self.region_paths = region_paths
        self.clip_feat_dim = map_config.visual_encoder.embedding_size
        self.force = force
        self.store_all = store_all

    def create_mobile_base_map(self):
        """
        build the 3D map centering at the first base frame
        """
        self.map_save_dir = self.data_dir / "vlmap"
        os.makedirs(self.map_save_dir, exist_ok=True)
        if self.store_all:
            self.map_save_path = self.map_save_dir / f"{self.map_config.map_prefix}-{self.map_config.cell_size}-alldata.h5df"
        else:
            self.map_save_path = self.map_save_dir / f"{self.map_config.map_prefix}-{self.map_config.cell_size}.h5df"

        if os.path.isfile(self.map_save_path):
            if not self.force:
                print(f"Map exists, exiting: {self.map_save_path})")
                exit(0)
            else:
                print("force = True, overwriting exiting map")
                os.remove(self.map_save_path)

        # access config info
        camera_height = self.map_config.pose_info.camera_height
        cs = self.map_config.cell_size
        gs = self.map_config.grid_size
        depth_sample_rate = self.map_config.depth_sample_rate

        self.base_poses = np.loadtxt(self.pose_path)
        self.init_base_tf = cvt_pose_vec2tf(self.base_poses[0])
        # print(self.init_base_tf)
        self.init_base_tf = (
            self.base_transform @ cvt_pose_vec2tf(self.base_poses[0]) @ np.linalg.inv(self.base_transform)
        )
        # print(self.init_base_tf)
        # tmp_trans = np.eye(4)
        # tmp_trans[:3, 3] = self.init_base_tf[:3, 3]
        # self.init_base_tf = self.base_transform @ (self.init_base_tf - tmp_trans) + tmp_trans
        self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        self.init_cam_tf = self.init_base_tf @ self.base2cam_tf
        self.inv_init_cam_tf = np.linalg.inv(self.init_cam_tf)

        # init pixelwise visual encoder
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        pixel_embedding_creator = PixelEmbeddingCreator(
            self.device,
            self.map_config.visual_encoder.name,
            self.map_config.visual_encoder.model_path,
            self.map_config.visual_encoder.embedding_size,
            self.map_config.visual_encoder.crop_size,
            self.map_config.visual_encoder.base_size,
        )

        # init the map
        (
            vh,
            grid_feat,
            grid_pos,
            weight,
            occupied_ids,
            grid_rgb,
            mapped_iter_set,
            max_id,
            grid_semantic,
            grid_region,
            grid_instance,
            grid_histogram
        ) = self._init_map(camera_height, cs, gs, self.map_save_path)

        # load camera calib matrix in config
        calib_mat = np.array(self.map_config.cam_calib_mat).reshape((3, 3))
        cv_map = np.zeros((gs, gs, 3), dtype=np.uint8)
        height_map = -100 * np.ones((gs, gs), dtype=np.float32)

        # semantic mapping
        semantics = defaultdict(lambda: defaultdict(int))
        regions = defaultdict(lambda: defaultdict(int))
        pbar = tqdm(
            zip(self.rgb_paths, self.depth_paths, self.base_poses, self.semantic_paths, self.region_paths),
            total=len(self.rgb_paths),
        )
        for frame_i, (rgb_path, depth_path, base_posevec, semantic_path, region_path) in enumerate(pbar):
            # load data
            habitat_base_pose = cvt_pose_vec2tf(base_posevec)
            base_pose = self.base_transform @ habitat_base_pose @ np.linalg.inv(self.base_transform)
            tf = self.inv_init_base_tf @ base_pose

            # theta = np.arctan2(tf[1, 0], tf[0, 0])
            # theta_deg = np.rad2deg(theta)
            # row, col, _ = base_pos2grid_id_3d(gs, cs, tf[0, 3], tf[1, 3], tf[2, 3])
            # trow, tcol, _ = base_pos2grid_id_3d(gs, cs, tf[0, 3] + tf[0, 0], tf[1, 3] + tf[1, 0], tf[2, 3])

            # cv2.circle(topdown, (col, row), 3, (0, 0, 255), -1)
            # cv2.circle(topdown, (tcol, trow), 3, (0, 255, 0), -1)

            bgr = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = load_depth_npy(depth_path)
            semantic = load_semantic_npy(semantic_path)
            region = load_semantic_npy(region_path)

            # get pixel-aligned features
            pix_feats = pixel_embedding_creator.get_image_features(rgb_path)
            pix_feats_intr = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

            # backproject depth point cloud
            pc = self._backproject_depth(depth, calib_mat, depth_sample_rate, min_depth=0.1, max_depth=6)

            # transform the point cloud to global frame (init base frame)
            # pc_transform = self.inv_init_base_tf @ self.base_transform @ habitat_base_pose @ self.base2cam_tf
            pc_transform = tf @ self.base_transform @ self.base2cam_tf
            pc_global = transform_pc(pc, pc_transform)  # (3, N)

            for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                row, col, height = base_pos2grid_id_3d(gs, cs, p[0], p[1], p[2])
                if self._out_of_range(row, col, height, gs, vh):
                    continue

                px, py, pz = project_point(calib_mat, p_local)
                rgb_v = rgb[py, px, :]
                semantic_v = semantic[py, px]
                region_v = region[py, px]
                px, py, pz = project_point(pix_feats_intr, p_local)

                if height > height_map[row, col]:
                    height_map[row, col] = height
                    cv_map[row, col, :] = rgb_v

                # when the max_id exceeds the reserved size,
                # double the grid_feat, grid_pos, weight, grid_rgb lengths
                if max_id >= grid_feat.shape[0]:
                    self._reserve_map_space(
                        grid_feat, grid_pos, weight, grid_rgb, grid_semantic, grid_region, grid_instance
                    )

                # apply the distance weighting according to
                # ConceptFusion https://arxiv.org/pdf/2302.07 241.pdf Sec. 4.1, Feature fusion
                # FIXME: here is the distance weighting
                distance_weighting = True
                if distance_weighting:
                    radial_dist_sq = np.sum(np.square(p_local))
                    # original value: sigma_sq = 0.6
                    sigma_sq = 0.6
                    alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))
                else:
                    # constant value
                    alpha = 1

                # update map features
                if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                    feat = pix_feats[0, :, py, px]
                    occupied_id = occupied_ids[row, col, height]

                    if self.store_all:
                        if occupied_id not in grid_histogram.keys():
                            grid_histogram[occupied_id] = []
                        lst = grid_histogram[occupied_id]
                        lst.append(feat.flatten())
                        grid_histogram[occupied_id] = lst

                    if occupied_id == -1:
                        occupied_ids[row, col, height] = max_id
                        grid_feat[max_id] = feat.flatten() * alpha
                        grid_rgb[max_id] = rgb_v

                        # semantics[max_id][semantic_v] += 1
                        # regions[max_id][region_v] += 1

                        grid_semantic[max_id] = semantic_v
                        grid_region[max_id] = region_v

                        weight[max_id] += alpha
                        grid_pos[max_id] = [row, col, height]
                        max_id += 1
                    else:
                        grid_feat[occupied_id] = (
                            grid_feat[occupied_id] * weight[occupied_id] + feat.flatten() * alpha
                        ) / (weight[occupied_id] + alpha)
                        grid_rgb[occupied_id] = (grid_rgb[occupied_id] * weight[occupied_id] + rgb_v * alpha) / (
                            weight[occupied_id] + alpha
                        )

                        # semantics[max_id][semantic_v] += (semantics[max_id][semantic_v] * weight[occupied_id] + 1 * alpha) / (
                        #     weight[occupied_id] + alpha
                        # )
                        # regions[max_id][region_v] += (regions[max_id][region_v] * weight[occupied_id] + 1 * alpha) / (
                        #     weight[occupied_id] + alpha
                        # )
                        grid_semantic[max_id] = (grid_semantic[occupied_id] * weight[occupied_id] + semantic_v * alpha) / (
                            weight[occupied_id] + alpha
                        )
                        grid_region[max_id] = (grid_region[occupied_id] * weight[occupied_id] + region_v * alpha) / (
                            weight[occupied_id] + alpha
                        )

                        # grid_semantic[occupied_id] = semantic_v
                        # grid_region[occupied_id] = region_v

                        # semantics[occupied_id][semantic_v] += 1
                        # regions[occupied_id][region_v] += 1

                        weight[occupied_id] += alpha

            mapped_iter_set.add(frame_i)
            #if frame_i % 25 == 24:
            #if frame_i % 100 == 99:
            if False:
                # if frame_i % 10 == 9:
                # grid_semantic = self.getHighestValueFromDict(semantics, grid_semantic)
                # grid_region = self.getHighestValueFromDict(regions, grid_region)
                print(f"Temporarily saving {max_id} features at iter {frame_i}...")
                self._save_3d_map(
                    grid_feat,
                    grid_pos,
                    weight,
                    grid_rgb,
                    occupied_ids,
                    mapped_iter_set,
                    max_id,
                    grid_semantic,
                    grid_region,
                    grid_instance,
                    grid_histogram
                )
        grid_semantic = self.getHighestValueFromDict(semantics, grid_semantic)
        grid_region = self.getHighestValueFromDict(regions, grid_region)
        self._save_3d_map(
            grid_feat,
            grid_pos,
            weight,
            grid_rgb,
            occupied_ids,
            mapped_iter_set,
            max_id,
            grid_semantic,
            grid_region,
            grid_instance,
            grid_histogram,
        )
        print("Saved map")

    def getHighestValueFromDict(self, input_dict, out):
        for max_id in input_dict:
            d = input_dict[max_id]
            if d:
                current_max_key = max(d, key=d.get)
                out[max_id] = current_max_key
            else:
                out[max_id] = 0
        return out

    def create_camera_map(self):
        """
        TODO: To be implemented
        build the 3D map centering at the first camera frame. We require that the camera is initialized
        horizontally (the optical axis is parallel to the floor at the first frame).
        """
        return NotImplementedError

    def _init_map(self, camera_height: float, cs: float, gs: int, map_path: Path) -> Tuple:
        """
        initialize a voxel grid of size (gs, gs, vh), vh = camera_height / cs, each voxel is of
        size cs
        """
        # init the map related variables
        vh = int(camera_height / cs)
        # FIXME: static map height
        #vh = int(7.5 / cs)
        grid_feat = np.zeros((gs * gs, self.clip_feat_dim), dtype=np.float32)
        grid_pos = np.zeros((gs * gs, 3), dtype=np.int32)
        occupied_ids = -1 * np.ones((gs, gs, vh), dtype=np.int32)
        weight = np.zeros((gs * gs), dtype=np.float32)
        grid_rgb = np.zeros((gs * gs, 3), dtype=np.uint8)
        mapped_iter_set = set()
        mapped_iter_list = list(mapped_iter_set)
        grid_semantic = np.zeros((gs * gs, 1), dtype=np.uint8)
        grid_region = np.zeros((gs * gs, 1), dtype=np.uint8)
        grid_instance = np.zeros((gs * gs, 1), dtype=np.uint16)
        grid_histogram = None
        if self.store_all:
            grid_histogram = {}
        max_id = 0

        # check if there is already saved map
        if os.path.exists(map_path):
            (
                mapped_iter_list,
                grid_feat,
                grid_pos,
                weight,
                occupied_ids,
                grid_rgb,
                grid_semantic,
                grid_region,
                grid_instance,
                grid_histogram
            ) = load_3d_map(self.map_save_path)
            mapped_iter_set = set(mapped_iter_list)
            max_id = grid_feat.shape[0]

        return (
            vh,
            grid_feat,
            grid_pos,
            weight,
            occupied_ids,
            grid_rgb,
            mapped_iter_set,
            max_id,
            grid_semantic,
            grid_region,
            grid_instance,
            grid_histogram
        )

    # def _init_openseg(self):
    #     checkpoint_path = self.map_config.visual_encoder.model_path
    #     print(f"Loading OpenSeg from {checkpoint_path}")
    #     # if not os.exists(checkpoint_path):
    #     #     print("Please download OpenSeg checkpoint.")
    #     #     return FileNotFoundError
    #     openseg_model, text_emb = init_openseg_model(checkpoint_path)
    #     self.clip_feat_dim = self.map_config.visual_encoder.embedding_size
    #     return openseg_model, text_emb

    # def _init_lseg(self):
    #     crop_size = 480  # 480
    #     base_size = 520  # 520
    #     if torch.cuda.is_available():
    #         self.device = "cuda"
    #     elif torch.backends.mps.is_available():
    #         self.device = "mps"
    #     else:
    #         self.device = "cpu"
    #     lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
    #     model_state_dict = lseg_model.state_dict()
    #     checkpoint_dir = Path(__file__).resolve().parents[1] / "lseg" / "checkpoints"
    #     checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
    #     print("checkpoint", checkpoint_path)
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     if not checkpoint_path.exists():
    #         print("Downloading LSeg checkpoint...")
    #         # the checkpoint is from official LSeg github repo
    #         # https://github.com/isl-org/lang-seg
    #         checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
    #         gdown.download(checkpoint_url, output=str(checkpoint_path))

    #     pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device)
    #     pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
    #     model_state_dict.update(pretrained_state_dict)
    #     lseg_model.load_state_dict(pretrained_state_dict)

    #     lseg_model.eval()
    #     lseg_model = lseg_model.to(self.device)

    #     norm_mean = [0.5, 0.5, 0.5]
    #     norm_std = [0.5, 0.5, 0.5]
    #     lseg_transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #         ]
    #     )
    #     self.clip_feat_dim = lseg_model.out_c
    #     return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std

    def _backproject_depth(
        self,
        depth: np.ndarray,
        calib_mat: np.ndarray,
        depth_sample_rate: int,
        min_depth: float = 0.1,
        max_depth: float = 10,
    ) -> np.ndarray:
        pc, mask = depth2pc(depth, intr_mat=calib_mat, min_depth=min_depth, max_depth=max_depth)  # (3, N)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        return pc

    def _out_of_range(self, row: int, col: int, height: int, gs: int, vh: int) -> bool:
        return col >= gs or row >= gs or height >= vh or col < 0 or row < 0 or height < 0

    def _reserve_map_space(
        self,
        grid_feat: np.ndarray,
        grid_pos: np.ndarray,
        weight: np.ndarray,
        grid_rgb: np.ndarray,
        grid_semantic: np.ndarray,
        grid_region: np.ndarray,
        grid_instance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid_feat = np.concatenate(
            [
                grid_feat,
                np.zeros((grid_feat.shape[0], grid_feat.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        grid_pos = np.concatenate(
            [
                grid_pos,
                np.zeros((grid_pos.shape[0], grid_pos.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        weight = np.concatenate([weight, np.zeros((weight.shape[0]), dtype=np.int32)], axis=0)
        grid_rgb = np.concatenate(
            [
                grid_rgb,
                np.zeros((grid_rgb.shape[0], grid_rgb.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        grid_semantic = np.concatenate(
            [
                grid_semantic,
                np.zeros((grid_semantic.shape[0], grid_semantic.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        grid_region = np.concatenate(
            [
                grid_region,
                np.zeros((grid_region.shape[0], grid_region.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        grid_instance = np.concatenate(
            [
                grid_instance,
                np.zeros((grid_instance.shape[0], grid_instance.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        return grid_feat, grid_pos, weight, grid_rgb, grid_semantic, grid_region, grid_instance

    def _save_3d_map(
        self,
        grid_feat: np.ndarray,
        grid_pos: np.ndarray,
        weight: np.ndarray,
        grid_rgb: np.ndarray,
        occupied_ids: Set,
        mapped_iter_set: Set,
        max_id: int,
        grid_semantic: np.ndarray,
        grid_region: np.ndarray,
        grid_instance: np.ndarray,
        grid_histogram: Optional[Dict]
    ) -> None:
        grid_feat = grid_feat[:max_id]
        grid_pos = grid_pos[:max_id]
        weight = weight[:max_id]
        grid_rgb = grid_rgb[:max_id]
        grid_semantic = grid_semantic[:max_id]
        grid_region = grid_region[:max_id]
        save_3d_map(
            self.map_save_path,
            grid_feat,
            grid_pos,
            weight,
            occupied_ids,
            list(mapped_iter_set),
            grid_rgb,
            grid_semantic,
            grid_region,
            grid_instance,
            grid_histogram,
        )
