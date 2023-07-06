from typing import Tuple, Union, Any
import numpy as np
import torch

SMPL_JOINT = 30
SMPL_JOINT_NAME = (
    'Pelvis', 'L_Hip', 'R_Hip', 
    'Torso', 'L_Knee', 'R_Knee', 
    'Spine', 'L_Ankle', 'R_Ankle', 
    'Chest', 'L_Toe', 'R_Toe', 
    'Neck', 'L_Thorax', 'R_Thorax',
    'Head', 'L_Shoulder', 'R_Shoulder', 
    'L_Elbow', 'R_Elbow', 
    'L_Wrist', 'R_Wrist', 
    'L_Hand', 'R_Hand', 
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 
    'Head_top'
)
SMPL_SKELETON = (
    (0,1), (1,4), (4,7), (7,10),
    (0,2), (2,5), (5,8), (8,11),
    (0,3), (3,6), (6,9), (9,14), (14,17), (17,19), (19, 21), (21,23),
    (9,13), (13,16), (16,18), (18,20), (20,22),
    (9,12), (12,24), (24,15),
    (24,25), (24,26), (25,27), (26,28), (24,29)
)

COCO_JOINT = 18
COCO_JOINT_NAME = (
    'Nose','Neck',
    'R_Shoulder','R_Elbow','R_Wrist',
    'L_Shoulder','L_Elbow','L_Wrist', 
    'R_Hip', 'R_Knee','R_Ankle',
    'L_Hip', 'L_Knee','L_Ankle', 
    'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
)
COCO_SKELETON = ( 
    ( 1,  2), ( 1,  5),             # Shoulder
    ( 2,  3), ( 3,  4),             # R_arm
    ( 5,  6), ( 6,  7),             # L_arm
    ( 1,  8), ( 8,  9), ( 9, 10),   # R_leg
    ( 1, 11), (11, 12), (12, 13),   # L_leg
    ( 1,  0),                       # Neck
    ( 0, 15), (15, 17),             # L_face
    ( 0, 14), (14, 16),             # R_face
)

COCO_COLORS = (
    (255,   0,   0), (255,  85,   0), (255, 170,   0), (255, 255,   0), (170, 255,   0),
    ( 85, 255,   0), (  0, 255,   0), (  0, 255,  85), (  0, 255, 170), (  0, 255, 255),
    (  0, 170, 255), (  0,  85, 255), (  0,   0, 255), ( 85,   0, 255), (170,   0, 255),
    (255,   0, 255), (255,   0, 170), (255,   0,  85)
)

def get_rotation_matrix(rot_x=0,rot_y=0,rot_z=0)->np.ndarray:
    # rot_y 90, rot_z=90
    rot_aug_mat_x = np.array([
        [                           1,                           0,                           0 ],
        [                           0,  np.cos(np.deg2rad(-rot_x)), -np.sin(np.deg2rad(-rot_x)) ],
        [                           0,  np.sin(np.deg2rad(-rot_x)),  np.cos(np.deg2rad(-rot_x)) ]],dtype=np.float32)
    
    rot_aug_mat_y = np.array([
        [  np.cos(np.deg2rad(-rot_y)),                           0,  np.sin(np.deg2rad(-rot_y)) ],
        [                           0,                           1,                           0 ],
        [ -np.sin(np.deg2rad(-rot_y)),                           0,  np.cos(np.deg2rad(-rot_y)) ]],dtype=np.float32)
    
    rot_aug_mat_z = np.array([
        [  np.cos(np.deg2rad(-rot_z)), -np.sin(np.deg2rad(-rot_z)),                           0 ],
        [  np.sin(np.deg2rad(-rot_z)),  np.cos(np.deg2rad(-rot_z)),                           0 ],
        [                           0,                           0,                           1 ]],dtype=np.float32)
    
    rot_mat=rot_aug_mat_x @ rot_aug_mat_y @ rot_aug_mat_z
    
    return rot_mat

def reduce_to_jointCOCO(joint:np.ndarray,torch_out=False):
    '''
    :param torch.Tensor joint: (30,3) JOINT
    :return torch.Tensor: (18,3) JOINT
    '''
    new_joint = []
    for name in COCO_JOINT_NAME:
        idx = SMPL_JOINT_NAME.index(name)
        new_joint.append(joint[idx,:])
    new_joint = np.stack(new_joint,1)
    if torch_out:
        new_joint = torch.Tensor(new_joint)
    return new_joint.T


def decompose_joint_dim(joint:Union[torch.Tensor,np.ndarray]):
    return joint[:,0], joint[:,1], joint[:,2]


ISOMETRIC_Z_PROJ = np.array([ [1,0], [0,1], [0,0] ])
def degenerate_joint(joint,transform_mat=ISOMETRIC_Z_PROJ) -> np.ndarray:
    '''
    degenerate joint3D -> joint2D\n
    or projection z-axis
    '''
    _joint = joint
    if type(joint) == torch.Tensor:
        _joint = joint.numpy()
    
    degenerated_joint = np.matmul(_joint,transform_mat)
    return degenerated_joint

def shift_joint(point: np.ndarray, move_vect=[0, 0]) -> np.ndarray:
    """point들를 move_vect로 평행이동함."""
    JOINT_NUM = point.shape[-2]
    JOINT_DIM = point.shape[-1]
    points = point.copy().reshape(-1, JOINT_DIM)

    if type(move_vect) != list:
        if type(move_vect) == np.ndarray:
            move_vect = move_vect.flatten().tolist()
        else:
            move_vect = list(move_vect)

    for units in range(JOINT_DIM):
        points[:, units] = points[:, units] + move_vect[units]

    result = points.reshape(-1, JOINT_NUM, JOINT_DIM)
    return result


def rotate_joint(point: np.ndarray, theta=0) -> np.ndarray:
    """2차원 회전 변환"""
    if not theta:
        return point

    points = point.copy()
    rot_mat = np.array(
        [
            [np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
            [np.sin(np.radians(theta)), np.cos(np.radians(theta))],
        ]
    )

    result = points @ rot_mat
    return result

def determine_cam_param(img_shape=[512,512],cam_from=4):
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, cam_from])


    def estimate_focal_length(img_h, img_w):
        return (img_w * img_w + img_h * img_h) ** 0.5
    focal_xy=estimate_focal_length(*img_shape)
    focal=[focal_xy,focal_xy]

    princ_pt=[int(img_shape[0]/2),int(img_shape[1]/2)]
    
    cam_param = (camera_pose, focal, princ_pt)
    return cam_param

def draw_joint_at_Pixel_moving_camera(joint3D, img_shape, n_frame,length,zoom,zoom_coef,trans,trans_coef,rotation,rotation_coef):
    # explain
    # zoom type => str only "in", "out"
    # zoom_coef => default 5
    # trans type => list [0,0] => [x,y]
    # trans_coef => list [1,2] => [x,2*y]
    # rotation type => list [0,0,0] => [x,y,z]
    # rotation_coef => [1.5,1.5,1.5] => [1.5*x,1.5*y,1.5*z]

    # this line is to apply rotation
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 4])
    if rotation != [0,0,0]:
        rotation, rotation_coef=np.array(rotation,dtype=np.uint8), np.array(rotation_coef,dtype=np.uint8)
        app_rotation=rotation*rotation_coef*n_frame
        rot_mat=get_rotation_matrix(app_rotation[0],app_rotation[1],app_rotation[2])
        camera_pose[:3,:3]=rot_mat

    # this line is to apply zoom
    def estimate_focal_length(img_h, img_w):
        return (img_w * img_w + img_h * img_h) ** 0.5
    focal_xy=estimate_focal_length(*img_shape)
    if zoom =="None":
        focal=[focal_xy,focal_xy]
    elif zoom =="in":
        app_zoom=(n_frame-length)*zoom_coef
        focal=[focal_xy+app_zoom,focal_xy+app_zoom]
    elif zoom =="out":
        app_zoom=(length-n_frame)*zoom_coef
        focal=[focal_xy+app_zoom,focal_xy+app_zoom]
    
    # this line is to apply trans
    if trans==[0,0]:
        princ_pt=[int(img_shape[0]/2),int(img_shape[1]/2)]
    else:
        trans,trans_coef=np.array(trans,dtype=np.uint8),np.array(trans_coef,dtype=np.uint8)
        app_trans=trans*(trans_coef * n_frame - length/2)
        princ_pt=[int(img_shape[0]/2)+app_trans[0],int(img_shape[1]/2)+app_trans[1]]

    img_x, img_y = img_shape
    joint_cam_coord=np.insert(joint3D,3,1,axis=1)
    joint_cam_coord=joint_cam_coord.transpose()
    joint_cam_coord=np.dot(camera_pose[:3],joint_cam_coord)
    joint_cam_coord=joint_cam_coord.transpose()
    
    x = joint_cam_coord[:,0] / joint_cam_coord[:,2] * focal[0] + princ_pt[0]
    y = joint_cam_coord[:,1] / joint_cam_coord[:,2] * focal[1] + princ_pt[1]
    y = img_y - y
    
    # z = cam_coord[:,2]
    return np.stack((x,y),1)

def draw_joint_at_Pixel(joint3D, img_shape, camera_pose, focal, princ_pt):
    img_x, img_y = img_shape
    joint_cam_coord=np.insert(joint3D,3,1,axis=1)
    joint_cam_coord=joint_cam_coord.transpose()
    joint_cam_coord=np.dot(camera_pose[:3],joint_cam_coord)
    joint_cam_coord=joint_cam_coord.transpose()
    
    x = joint_cam_coord[:,0] / joint_cam_coord[:,2] * focal[0] + princ_pt[0]
    y = joint_cam_coord[:,1] / joint_cam_coord[:,2] * focal[1] + princ_pt[1]
    y = img_y - y
    
    # z = cam_coord[:,2]
    return np.stack((x,y),1)


""" 
def draw_jointCOCO(joint3D, img_shape):
    jointPix = draw_joint_at_Pixel(joint3D,img_shape,*determine_cam_param(img_shape))
    jointPixCOCO = reduce_to_jointCOCO(jointPix)
    return jointPixCOCO """


import math
import cv2
def vis_coco_skeleton(joint2D, skeleton=COCO_SKELETON, COLOR=COCO_COLORS,np_img=np.zeros((512,512,3)))->np.ndarray:
    line_thick = 3
    circle_radius = 3
    joint2D = joint2D.astype(np.int32)

    img = np.copy(np_img)
    for i, (x,y) in enumerate(joint2D):
        cv2.circle(img, (x,y), circle_radius, COLOR[i], thickness=-1)
    
    for i, joint_idx in enumerate(skeleton):
        cur_canvas = img.copy()
        
        Y, X = joint2D[joint_idx,:].T
        mX = np.mean(X)
        mY = np.mean(Y)
        
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), line_thick), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(img, polygon, COLOR[i])
        
        img = cv2.addWeighted(img, 0.8, cur_canvas, 0.2, 0)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.flip(img,1)

    return img

def write_image(file_name,img:np.ndarray)->None:
    cv2.imwrite(f"{file_name}.png",img)



##################################################TEMP###############

"""
SMPL_SKELETON_TRANSPOSED = [
    [0, 1, 4,  7, 0, 2, 5,  8, 0, 3, 6,  9, 14, 17, 19, 21,  9, 13, 16, 18, 20,  9, 12, 24, 24, 24, 25, 26, 24],
    [1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 14, 17, 19, 21, 23, 13, 16, 18, 20, 22, 12, 24, 15, 25, 26, 27, 28, 29]
]
SKELETON_LINE_START, SKELETON_LINE_END = SMPL_SKELETON_TRANSPOSED
"""