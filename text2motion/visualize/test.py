import logging as log
import cv2
text_lists = None
camera_perspective = None
last_generated = ""
category_selected = None
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",type=str, required=True, help="Category Select"
    )
    parser.add_argument(
        "--text_file", type=str, default="./output_categorized/text_desc_loc.json", help="text to be encoded"
    )
    parser.add_argument(
        "--state", type=str, default="", help="last generated prompt - 안넣으면 처음부터함." 
    )

    args = parser.parse_args()

    # change the text here
    text_lists = args.text_file
    last_generated = args.state
    category_selected =args.tag
    CATEGORY = ["jump", "run","climb", "throw","kick", "punch", "clap", "pick", "golf", "sit"]
    assert category_selected in CATEGORY, f"category must be {CATEGORY}, not {category_selected}"
        

from tqdm import tqdm
import sys
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
# os.system('pip install ./pyrender')

sys.argv = ["./VQ-Trans/GPT_eval_multi.py"]
sys.path.append("./VQ-Trans")
sys.path.append("./pyrender")

# os.chdir('./VQ-Trans')
import options.option_transformer as option_trans

args = option_trans.get_args_parser()

args.dataname = "t2m"
model_path = "../T2M-GPT"
args.resume_pth = f"{model_path}/VQVAE/net_last.pth"
args.resume_trans = f"{model_path}/VQTransformer_corruption05/net_best_fid.pth"
args.down_t = 2
args.depth = 3
args.block_size = 51

import clip
import torch
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans
from utils.motion_process import recover_from_ric
from models.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import gc

import torch
from visualize.simplify_loc2rot import joints2smpl

## load clip model and datasets
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(device)
clip_model, clip_preprocess = clip.load(
    "ViT-B/32", device=device, jit=False, download_root="./"
)  # Must set jit=False for training

if is_cuda:
    clip.model.convert_weights(clip_model)

clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(
    args,  ## use args to define different parameters in different quantizers
    args.nb_code,
    args.code_dim,
    args.output_emb_width,
    args.down_t,
    args.stride_t,
    args.width,
    args.depth,
    args.dilation_growth_rate,
)

trans_encoder = trans.Text2Motion_Transformer(
    num_vq=args.nb_code,
    embed_dim=1024,
    clip_dim=args.clip_dim,
    block_size=args.block_size,
    num_layers=9,
    n_head=16,
    drop_out_rate=args.drop_out_rate,
    fc_rate=args.ff_rate,
)


print("loading checkpoint from {}".format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location="cpu")
net.load_state_dict(ckpt["net"], strict=True)
net.eval()

print("loading transformer checkpoint from {}".format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location="cpu")
trans_encoder.load_state_dict(ckpt["trans"], strict=True)
trans_encoder.eval()

mean = torch.from_numpy(np.load(f"{model_path}/meta/mean.npy"))
std = torch.from_numpy(np.load(f"{model_path}/meta/std.npy"))

if is_cuda:
    net.cuda()
    trans_encoder.cuda()
    mean = mean.cuda()
    std = std.cuda()


def render(motions, name="test_vis"):
    define_rotation_factor=0.2
    # motion.shape (frame, 22, 3)
    save_name = name.replace(" ","_")

    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    is_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if is_cuda else "cpu")
    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=is_cuda)
    rot2xyz = Rotation2xyz(device=device)
        
    rot_x = np.clip(define_rotation_factor, -2.0,
                    2.0)*180
    rot_y = np.clip(define_rotation_factor, -2.0,
                    2.0)*180
    rot_z = np.clip(define_rotation_factor, -2.0,
                    2.0)*180

    rot_aug_mat_x = np.array([[1,0,0],
                            [0,np.cos(np.deg2rad(-rot_x)),-np.sin(np.deg2rad(-rot_x))],
                            [0,np.sin(np.deg2rad(-rot_x)),np.cos(np.deg2rad(-rot_x))]], dtype=np.float32)

    rot_aug_mat_y = np.array([[np.cos(np.deg2rad(-rot_y)),0,np.sin(np.deg2rad(-rot_y))],
                            [0,1,0],
                            [-np.sin(np.deg2rad(-rot_y)),0,np.cos(np.deg2rad(-rot_y))]], dtype=np.float32)

    rot_aug_mat_z = np.array([[np.cos(np.deg2rad(-rot_z)), -np.sin(np.deg2rad(-rot_z)), 0],
                                [np.sin(np.deg2rad(-rot_z)), np.cos(np.deg2rad(-rot_z)), 0],
                                [0, 0, 1]], dtype=np.float32)
    rot_aug_mat=rot_aug_mat_x @ rot_aug_mat_y @ rot_aug_mat_z

    motion_root_pose=motions[:,0,:]
    
    # just test
    rod_list=[]
    for i in motion_root_pose:
        root_pose, _ = cv2.Rodrigues(i)
        rod_list.append(root_pose)
    new_root_pose=[]
    for i in rod_list:
        root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,i))
        new_root_pose.append(root_pose)
    new_root_pose=np.array(new_root_pose)
    motions[:,0,:]=new_root_pose
    print(f"Running SMPLify, it may take a few minutes.")
    motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

    vertices = rot2xyz(
        torch.tensor(motion_tensor).clone().detach(),
        mask=None,
        pose_rep="rot6d",
        translation=True,
        glob=True,
        jointstype="vertices",
        vertstrans=True,
    )
    vertices = vertices.detach().cpu()
    torch.save(vertices, f"output_categorized/{category_selected}/{save_name}.pt")


def predict(clip_text):
    mesh_save_name = clip_text.replace(" ", "_")
    
    gc.collect()
    print("prompt text instruction: {}".format(clip_text))
    if torch.cuda.is_available():
        text = clip.tokenize([clip_text], truncate=True).cuda()
    else:
        text = clip.tokenize([clip_text], truncate=True)
    feat_clip_text = clip_model.encode_text(text).float()
    index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
    pred_pose = net.forward_decoder(index_motion)
    pred_xyz = recover_from_ric((pred_pose * std + mean).float(), 22)
    # output_name = hashlib.md5(clip_text.encode()).hexdigest()

    render(
        pred_xyz.detach().cpu().numpy().squeeze(axis=0),
        name=mesh_save_name,
    )


if __name__ == "__main__" and text_lists:
    import json
    with open(text_lists, 'r') as f:
        prompts_JSON = json.load(f)    
    for _, prompt in enumerate([ prompts_JSON[0] ]):
        predict(prompt)
    """ 
    start_idx = 0
    which_state_saved = 0
    
    if "" != last_generated:
        with open(last_generated,"r") as f:
            last_state = json.load(f)
            
        which_state_saved = int(last_generated.split(".")[0].split("_")[-1])
        assert prompts_JSON[ last_state["index"] ] == last_state["prompt"], "Not Accurate State"
        start_idx = last_state["index"] + 1
    else:
        print("[WARN] It'll start at first")
    
    which_state = text_lists.split(".")[0]
    if os.path.exists(f"{which_state}_LastIndex_{which_state_saved:04d}.json"):
        which_state_saved+=1
        
    for idx, prompt in enumerate(prompts_JSON[start_idx:]):
        prompt__ = prompt.replace(" ", "_")
        if os.path.exists(f"output_categorized/{category_selected}/{prompt__}.pt"):
            print(f"{prompt__}.pt is already exist")
            continue
        if (category_selected in prompt) and (category_selected != None):          
            with open(f"{which_state}_{category_selected}LastIndex_{which_state_saved:04d}.json",'w') as f:
                state = { "prompt" : prompt, "index" : start_idx+idx }
                json.dump(state,f)
            predict(clip_text=prompt)
        else:
            continue
    print("DONE") """