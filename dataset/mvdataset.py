import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from kiui.op import safe_normalize

import numpy as np
import cv2
import pickle
import zipfile


def get_rays(pose, h, w, fovy, opengl=True):
    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d


class MVDataset_code_t5_cam(Dataset):
    def __init__(self, args):
        super().__init__()
        with open(args.obj_pkl_path, 'rb') as f:
            self.obj_list = pickle.load(f)

        self.image_code_dir = args.image_code_path
        self.t5_embedding_dir = args.t5_feat_path
        self.camera_pose_dir = args.camera_pose_path

        self.cls_token_num = args.cls_token_num
        self.image_code_token = int(args.image_size // args.downsample_size) ** 2

        # self.train_index = np.random.permutation(np.arange(self.paths[index]['view_num']))[:8]
        self.train_index = [0, 2, 4, 6, 8, 10, 12, 14] # [0, 4, 8, 12]

    def __getitem__(self, index):
        obj_id = self.obj_list[index]
        image_code_file = os.path.join(self.image_code_dir, obj_id + ".npy")
        t5_file = os.path.join(self.t5_embedding_dir, obj_id + ".npy")
        camera_pose_file = os.path.join(self.camera_pose_dir, obj_id + ".npy")

        # 1. Load image code.
        image_code = torch.from_numpy(np.load(image_code_file)).long()  # T, N
        image_code = image_code[self.train_index, :]

        # 2. Load T5 embedding.
        t5_feat_padding = torch.zeros((self.cls_token_num, 2048))
        t5_feat = torch.from_numpy(np.load(t5_file)).squeeze(0)
        feat_len = min(self.cls_token_num, t5_feat.shape[0])
        t5_feat_padding[-feat_len:, :] = t5_feat[:feat_len, :]

        # 3. Load camera pose.
        camera_pose = torch.from_numpy(np.load(camera_pose_file))  # T, N
        camera_pose = camera_pose[self.train_index, :]

        # 4. Generation attention mask.
        emb_mask = torch.zeros((self.cls_token_num,))
        emb_mask[-feat_len:] = 1
        attn_mask = torch.tril(torch.ones(self.cls_token_num + self.image_code_token + 16 + 1, self.cls_token_num + self.image_code_token + 16 + 1))
        T = self.cls_token_num
        attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
        eye_matrix = torch.eye(self.cls_token_num + self.image_code_token + 16 + 1, self.cls_token_num + self.image_code_token + 16 + 1)
        attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
        attn_mask = attn_mask.to(torch.bool)
        attn_mask = attn_mask.reshape(1, attn_mask.shape[-2], attn_mask.shape[-1]).repeat(camera_pose.shape[0], 1, 1)

        # 5. Define valid.
        valid = torch.tensor(1)
        return image_code, t5_feat_padding, camera_pose, attn_mask, valid

    def __len__(self):
        return len(self.obj_list)


class NVDataset_image_t5_cam(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        with open(args.obj_pkl_path, 'rb') as f:
            self.obj_list = pickle.load(f)

        self.root_dir = args.root_dir
        self.t5_embedding_dir = args.t5_feat_path
        self.camera_pose_dir = args.camera_pose_path
        self.num_viewes = args.num_viewes

        self.cls_token_num = args.cls_token_num
        self.image_code_token = int(args.image_size // args.downsample_size) ** 2

        # self.train_index = np.random.permutation(np.arange(self.paths[index]['view_num']))[:8]
        self.train_index = [0, 2, 4, 6, 8, 10, 12, 14]
    
    def __get_image(self, img_path, zpfile=None):
        """
        Read img from path
        Args:
            img_path: path of img
        Return:
            img[np.array]: image in range[0, 1]
            mask[np.array]: image foreground mask
        """
        imgbytes = zpfile.read(img_path)
        img = cv2.imdecode(np.frombuffer(imgbytes, np.uint8),
                            cv2.IMREAD_UNCHANGED)
        # TODO
        img = cv2.resize(img, (self.args.image_size, self.args.image_size))
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[2] == 4:
            mask = img[..., 3:] * 1.0
        else:
            mask = np.ones_like(img) * 1.0

        # if self.replace_bg:
        #     img = img[..., :3] * img[..., 3:] + \
        #         self.background_color * (1 - img[..., 3:])
        # else:
        img = img[..., :3]
        return img, mask

    def __getitem__(self, index):
        obj_id = self.obj_list[index]

        t5_file = os.path.join(self.t5_embedding_dir, obj_id + ".npy")

        # 1. Load images.
        zpfile = zipfile.ZipFile(os.path.join(self.root_dir, obj_id + '.zip'))
        images = []
        for index in self.train_index:
            image, _ = self.__get_image(os.path.join(obj_id, "%03d.png" % index),
                                     zpfile=zpfile)
            images.append(torch.from_numpy(image))
        images = torch.stack(images)
        zpfile.close()

        # 2. Load T5 embedding.
        t5_feat_padding = torch.zeros((self.cls_token_num, 2048))
        t5_feat = torch.from_numpy(np.load(t5_file)).squeeze(0)
        feat_len = min(self.cls_token_num, t5_feat.shape[0])
        t5_feat_padding[-feat_len:, :] = t5_feat[:feat_len, :]

        # 3. Load camera pose.
        camera_pose_file = os.path.join(self.camera_pose_dir, obj_id + ".npy")
        camera_poses = torch.from_numpy(np.load(camera_pose_file))  # T, N
        camera_poses = camera_poses[self.train_index, :]
        rays_plucker = []
        for camera_pose in camera_poses:
            rays_o, rays_d = get_rays(camera_pose.float(), self.args.image_size, self.args.image_size, 49.1)
            ray_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
            rays_plucker.append(ray_plucker)
        rays_plucker = torch.stack(rays_plucker)

        # 4. Generation attention mask.
        emb_mask = torch.zeros((self.cls_token_num,))
        emb_mask[-feat_len:] = 1
        attn_mask = torch.tril(torch.ones(self.cls_token_num + self.image_code_token * self.num_viewes + 1, self.cls_token_num + self.image_code_token * self.num_viewes + 1))
        T = self.cls_token_num
        attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
        eye_matrix = torch.eye(self.cls_token_num + self.image_code_token * self.num_viewes + 1, self.cls_token_num + self.image_code_token * self.num_viewes + 1)
        attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
        attn_mask = attn_mask.to(torch.bool)
        attn_mask = attn_mask.reshape(1, attn_mask.shape[-2], attn_mask.shape[-1])

        # 5. Define valid.
        valid = torch.tensor(1, dtype=torch.float32)
        return images, t5_feat_padding, rays_plucker, attn_mask, valid

    def __len__(self):
        return len(self.obj_list)


def build_t_cam2i_code(args):
    return MVDataset_code_t5_cam(args)


def build_t_ray2i_nv(args):
    return NVDataset_image_t5_cam(args)
