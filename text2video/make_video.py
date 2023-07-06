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
    pass
    