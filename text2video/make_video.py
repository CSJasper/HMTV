import torch
from model import Model
import os
import argparse

def save_video(
        text_prompt,
        skeleton_path,
        output_root
):
    model = Model(device='cuda', dtype=torch.float16)
    model.process_controlnet_pose(text_prompt, prompt=text_prompt, save_path=os.path.join(output_root, text_prompt, "*.mp4"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True, help='output root')
    parser.add_argument("--spath", type=str, required=True, help= 'input skeleton video path')
    parser.add_argument("--text_prompt", type=str, required=True, help='text prompt')

    args = parser.parse_args()

    save_video(args.text_prompt, args.spath, args.text_prompt)

    