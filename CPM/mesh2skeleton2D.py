import os.path as osp
import os
from tqdm import tqdm
from glob import glob
import argparse
from smplpytorch.pytorch.smpl_layer import SMPL_Layer as SMPL
import cv2
import numpy as np
import torch
import SkeletonTrans as skel
import sys
import json
from direction import DIRECTION_LOOKUP_TABLE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int , default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--p_cam", required=True, type=str, default="", help='''
    Supported camera prompts :
    zoom_in
    zoom_out
    translation_x
    translation_y
    rotation_x
    rotation_y
    rotation_z
    zoom_in_rotation_y
    None
    If p_cam None you should specify arguments
    ''')
                            
    parser.add_argument("--json_path", type=str, required=True, help='json path for prompt-mesh pairs')
    parser.add_argument("--dist", type=int, default=4)
    parser.add_argument("--zoom", type=str, default=None),
    parser.add_argument("--zoom_coef", type=int, default=5)
    parser.add_argument("--trans", type=list, default=[0,0])
    parser.add_argument("--trans_coef", type=list, default=[2,2])
    parser.add_argument("--rotation", type=list, default=[0,0,0])
    parser.add_argument("--rotation_coef", type=list, default=[1.5,1.5,1.5])
    parser.add_argument("--output", type=str, required=True, help='output skeleton root path')
    parser.add_argument("--save_3d", action='store_true')

    args = parser.parse_args()

    H = int(args.height)
    W = int(args.width)
    image_shape_ = (H, W)
    json_path = args.json_path
    out_root = args.output
    p_cam_ = args.p_cam

###### if __name__ is __main__  ########

sys.path.insert(0, '../text2mesh/body_models')
current_dir = osp.abspath('.')

skeleton_path = './skeletons'

transform_SMPL2COCO = skel.reduce_to_jointCOCO
SMPL_SKELETON = skel.SMPL_SKELETON
COCO_SKELETON = skel.COCO_SKELETON
COCO_COLORS = skel.COCO_COLORS


def mesh2skeletons(
        mesh_path_or_mesh,
        save_name: str,
        image_shape: tuple,
        p_cam_or_none,
        output_root:str,
        save_3d: bool
):
    if isinstance(mesh_path_or_mesh, str):
        motion_mesh = torch.load(mesh_path_or_mesh)
    else:
        motion_mesh = mesh_path_or_mesh
    motions = motion_mesh[0].permute(2, 0, 1)
    shrink_factor = 1

    motions = motions[::shrink_factor, :, :]  # step size = shrink_factor
    j_regressor = SMPL(gender='neutral').th_J_regressor.numpy()
    face_kps_vertext = (331, 2802, 6262, 3489, 3990)
    nose_onehot = np.array([1 if i == 331 else 0 for i in range(j_regressor.shape[1])], dtype=np.float32).reshape((1, -1))
    left_eye_onehot = np.array([1 if i == 2802 else 0 for i in range(j_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
    right_eye_onehot = np.array([1 if i == 6262 else 0 for i in range(j_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
    left_ear_onehot = np.array([1 if i == 3489 else 0 for i in range(j_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
    right_ear_onehot = np.array([1 if i == 3990 else 0 for i in range(j_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
    j_regressor = np.concatenate((j_regressor, nose_onehot, left_eye_onehot, right_eye_onehot, left_ear_onehot, right_ear_onehot))
    j_regressor_extra = np.load('/database/jasper/wacv2024/codes/text2mesh/body_models/smpl/J_regressor_extra.npy')
    joint_regressor = np.concatenate((j_regressor, j_regressor_extra[3:4, :])).astype(np.float32)

    J_reg = torch.Tensor(joint_regressor)

    motions = motion_mesh[0].permute(2, 0, 1)
    joint3D_smpl = J_reg @ motions
    joint3D_smpl = joint3D_smpl.numpy()
    
    if save_3d:
        os.makedirs('./3djoints', exist_ok=True)
        np.save(f'./{output_root}/{save_name}.npy', joint3D_smpl)

    NB_FRAME, NB_JOINT, NB_DIM = tuple(joint3D_smpl.shape)
    # print(jointsmpl.shape)

    joint3D_COCO = np.zeros((NB_FRAME, skel.COCO_JOINT, NB_DIM))
    for i, joint3D in enumerate(joint3D_smpl):
        joint3D_COCO[i, :, :] = transform_SMPL2COCO(joint3D)
    # print(joint3D_COCO.shape)

    ###################################
    # projection
    img = np.ones((*image_shape, 3)) * 0
    length=len(joint3D_COCO)
    #cam_param = skel.determine_cam_param(image_shape,cam_from=dist)
    jointPix_COCO = [
        skel.draw_joint_at_Pixel_moving_camera(frame, image_shape, n,length,**DIRECTION_LOOKUP_TABLE[p_cam_or_none]) if p_cam_or_none is not None else skel.draw_joint_at_Pixel_moving_camera(frame, image_shape, n, length, "None", 5, [0,0], [2, 2], [0, 0, 0], [1.5, 1.5, 1.5])
        for n,frame in enumerate(joint3D_COCO)
    ]
    image_COCO = [
        skel.vis_coco_skeleton(j, COCO_SKELETON, np_img=img).astype(np.uint8)
        for j in jointPix_COCO
    ]

    video_name = osp.join(output_root, f"{save_name}.mp4")

    video_writer = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc(*"mp4v"), 20, image_COCO[0].shape[:2]
    )

    print('making video')

    for frame in tqdm(image_COCO):
        video_writer.write(frame)
    video_writer.release()

    print('video saved')

if __name__ == '__main__':
    save_dict = dict()
    with open(json_path, 'r') as jf:
        prompt_meshpath = json.load(jf)

    os.makedirs(out_root, exist_ok=True)

    for prompt, mesh_path in prompt_meshpath.items():
        mesh_path = mesh_path[2:]
        mesh_path = os.path.join('../', "text2mesh", mesh_path)
        mesh2skeletons(mesh_path, prompt, image_shape_, p_cam_, out_root, args.save_3d)

    print("Done")
        

