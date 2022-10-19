"""
Create an ImageDataset on the fly from a collections of images with corresponding OpenPose detections.
In order to preserve backwards compatibility with SMPLify-X, parts of the code are adapted from
https://github.com/vchoutas/smplify-x/blob/master/smplifyx/data_parser.py
"""
import os
import json
from pathlib import Path
import numpy as np
import re
from typing import Dict, Optional
from tqdm import tqdm

from yacs.config import CfgNode

from .image_dataset import ImageDataset
from ..utils.rotation_conversions import matrix_to_axis_angle as rotmat_to_aa_torch

def rotmat_to_aa(rotmat):
    import torch
    return rotmat_to_aa_torch(torch.from_numpy(rotmat)).numpy()

def regex_to_matcher(regexp):
    if regexp.startswith('__lex__'):
        start, end = regexp[len('__lex__'):].split(',')
        print(f'Will run on all names between {start} and {end}')
        return lambda x: (start <= x <= end)
    else:
        print(f'Will run on all names matching {regexp}')
        __pattern = re.compile(regexp)
        return lambda x: __pattern.match(x)

def read_openpose(keypoint_fn: str, max_people_per_image: Optional[int] = None):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    body_keypoints_2d = []

    for i, person in enumerate(data['people']):
        if max_people_per_image is not None and i >= max_people_per_image:
            break
        openpose_detections = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
        body_keypoints_2d.append(openpose_detections)

    return body_keypoints_2d

class OpenPoseDataset(ImageDataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_folder: str,
                 keypoint_folder: str,
                 rescale_factor: float = 1.2,
                 train: bool = False,
                 max_people_per_image: Optional[int] = None,
                 img_name_filter: str = '.*',
                 prohmr_fits_folder: Optional[str] = None,
                 walk_subdirectories: bool = False,
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations from image/OpenPose pairs.
        It builds and ImageDataset on-the-fly instead of reading the data from an npz file.
        Args:
            cfg (CfgNode): Model config file.
            img_folder (str): Folder containing images.
            keypoint_folder (str): Folder containing OpenPose detections.
            rescale_factor (float): Scale factor for rescaling bounding boxes computed from the OpenPose keypoints.
            train (bool): Whether it is for training or not (enables data augmentation)
        """

        super(ImageDataset, self).__init__()
        self.cfg = cfg
        self.img_folder = img_folder
        self.keypoint_folder = keypoint_folder
        self.prohmr_fits_folder = prohmr_fits_folder

        matcher = regex_to_matcher(img_name_filter)
        if not walk_subdirectories:
            self.img_paths = [os.path.join(self.img_folder, img_fn)
                            for img_fn in os.listdir(self.img_folder) if matcher(img_fn)]
        else:
            self.img_paths = []
            for root, _, files in os.walk(self.img_folder):
                for img_fn in files:
                    img_fpath = os.path.join(root, img_fn)
                    img_fn_relative = os.path.relpath(img_fpath, self.img_folder)
                    if matcher(img_fn_relative):
                        self.img_paths.append(img_fpath)
        print(f'Found {len(self.img_paths)} images in {self.img_folder}')
        self.img_paths = sorted(self.img_paths)
        self.rescale_factor = rescale_factor
        self.train = train
        self.max_people_per_image = max_people_per_image
        self.img_dir = self.img_folder
        body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
        extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
        flip_keypoint_permutation = body_permutation + [25 + i for i in extra_permutation]
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)
        self.flip_keypoint_permutation = flip_keypoint_permutation
        self.preprocess()

    def preprocess(self):
        """
        Preprocess annotations and convert them to the format ImageDataset expects.
        """
        body_keypoints = []
        imgnames = []
        personids = []  # Monitor id of person within image
        scales = []
        centers = []
        body_pose = []
        has_body_pose = []
        betas = []
        has_betas = []
        extra_info = []
        num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)
        PSEUDO_GT_EXISTS = self.prohmr_fits_folder is not None and os.path.exists(self.prohmr_fits_folder)
        for i in tqdm(range(len(self.img_paths))):
            img_path = self.img_paths[i]
            item = self.get_example(img_path)
            if len(item) == 0:
                continue
            num_people = item['keypoints_2d'].shape[0]
            for n in range(num_people):
                keypoints_n = item['keypoints_2d'][n]
                keypoints_valid_n = keypoints_n[keypoints_n[:, 1] > 0, :].copy()
                bbox = [min(keypoints_valid_n[:,0]), min(keypoints_valid_n[:,1]),
                    max(keypoints_valid_n[:,0]), max(keypoints_valid_n[:,1])]
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                scale = self.rescale_factor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])
                body_keypoints.append(keypoints_n)
                scales.append(scale)
                centers.append(center)
                assert os.path.join(self.img_dir, item['img_name']) == img_path
                imgnames.append(item['img_name'])
                personids.append(n)

                img_fn, img_ext = os.path.splitext(item['img_name'])
                prohmr_fit_path = f'{self.prohmr_fits_folder}/{img_fn}_p{n}_fitting.npz' # Assuming fixed name format for pseudo-gt fitting npz
                if PSEUDO_GT_EXISTS and Path(prohmr_fit_path).exists():
                    prohmr_fit = np.load(prohmr_fit_path, allow_pickle=True)
                    body_pose_rotmat = np.concatenate([prohmr_fit['global_orient'][None], prohmr_fit['body_pose']], axis=0)
                    body_pose_rotmat_aa = rotmat_to_aa(body_pose_rotmat)
                    assert body_pose_rotmat_aa.shape == (self.cfg.SMPL.NUM_BODY_JOINTS + 1, 3)
                    body_pose.append(body_pose_rotmat_aa.reshape(-1))
                    has_body_pose.append(1)
                    betas.append(prohmr_fit['betas'])
                    has_betas.append(1)
                    extra_info.append({
                        'fitting_loss': prohmr_fit['losses'],
                        'fitting_cam_t': prohmr_fit['camera_translation']
                    })
                else:
                    body_pose.append(np.zeros(num_pose, dtype=np.float32))
                    has_body_pose.append(0)
                    betas.append(np.zeros(10, dtype=np.float32))
                    has_betas.append(0)
                    extra_info.append({})

        self.imgname = np.array(imgnames)
        self.personid = np.array(personids, dtype=np.int32)
        self.scale = np.array(scales).astype(np.float32) / 200.0
        self.center = np.array(centers).astype(np.float32)
        body_keypoints_2d = np.array(body_keypoints).astype(np.float32)
        N = len(self.center)
        extra_keypoints_2d = np.zeros((N, 19, 3))
        self.keypoints_2d = np.concatenate((body_keypoints_2d, extra_keypoints_2d), axis=1).astype(np.float32)
        body_keypoints_3d = np.zeros((N, 25, 4), dtype=np.float32)
        extra_keypoints_3d = np.zeros((N, 19, 4), dtype=np.float32)
        self.keypoints_3d = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1).astype(np.float32)

        self.body_pose = np.stack(body_pose, axis=0).astype(np.float32)
        self.has_body_pose = np.array(has_body_pose, dtype=np.float32)
        self.betas = np.stack(betas, axis=0).astype(np.float32)
        self.has_betas = np.array(has_betas, dtype=np.float32)
        assert self.body_pose.shape == (N, num_pose), self.body_pose.shape
        assert self.has_body_pose.shape == (N,)
        assert self.betas.shape == (N, 10)
        assert self.has_betas.shape == (N,)
        self.extra_info = extra_info

    def get_example(self, img_path: str) -> Dict:
        """
        Load an image and corresponding OpenPose detections.
        Args:
            img_path (str): Path to image file.
        Returns:
            Dict: Dictionary containing the image path and 2D keypoints if available, else an empty dictionary.
        """
        # img_name = os.path.split(img_path)[1]
        assert Path(self.img_dir) in Path(img_path).parents
        img_name = os.path.relpath(img_path, self.img_dir)
        img_fn, _ = os.path.splitext(img_name)

        keypoint_fn = os.path.join(self.keypoint_folder,
                               img_fn + '_keypoints.json')
        keypoints_2d = read_openpose(keypoint_fn, max_people_per_image=self.max_people_per_image)

        if len(keypoints_2d) < 1:
            return {}
        keypoints_2d = np.stack(keypoints_2d)

        item = {'img_path': img_path,
                'img_name': img_name,
                'keypoints_2d': keypoints_2d}
        return item

if __name__ == '__main__':
    from prohmr.configs import prohmr_config

    ROOT='/home/shubham/code/stable-humans/logs_/dev/insta/'
    IMAGES_DIR=f"{ROOT}/images/instavariety/"
    DETECTIONS_DIR=f"{ROOT}/vitdet/instavariety/"
    KEYPOINTS_DIR=f"{ROOT}/vitpose/instavariety/"
    PROHMR_FIT_DIR=f"{ROOT}/prohmr_fit/instavariety/"

    model_cfg = prohmr_config()

    dataset = OpenPoseDataset(
        cfg=model_cfg,
        img_folder=IMAGES_DIR,
        keypoint_folder=KEYPOINTS_DIR,
        prohmr_fits_folder=PROHMR_FIT_DIR,
    )

    print(dataset.has_betas.mean())
    print(dataset.has_body_pose.mean())
    print(dataset.betas.mean(axis=0))
    print(dataset.body_pose.mean(axis=0))

    dataset2 = OpenPoseDataset(
        cfg=model_cfg,
        img_folder=IMAGES_DIR,
        keypoint_folder=KEYPOINTS_DIR,
    )
    print(dataset2.has_betas.mean())
    print(dataset2.has_body_pose.mean())
