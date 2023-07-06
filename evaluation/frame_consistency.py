import argparse
import clip
import torch
import numpy as np
import os
import json
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.nn.functional import cosine_similarity
import cv2
from PIL import Image
import os.path as osp
parser = argparse.ArgumentParser()
parser.add_argument("--vpath", type=str)

args = parser.parse_args()

def get_frame_consistency(vpath: str):
    video_path = vpath
    video_list=os.listdir(video_path)
    result_list=[]
    for video_name in tqdm(video_list):
        video = cv2.VideoCapture(osp.join(video_path,video_name))
        frames = []
        while True:
            success, img = video.read()
            if not success:
                break
                
            frames.append(img)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


        #text = clip.tokenize(raw_text.split()).to(device)

        preprocessed_img = []

        for f in frames:
            f = Image.fromarray(f)
            preprocessed_img.append(preprocess(f).unsqueeze(0).to(device))
        
        img_features = []
        with torch.no_grad():
            for ppim in preprocessed_img:
                img_features.append(model.encode_image(ppim))
        
        avg = 0

        for i in range(len(img_features) - 1):
            avg += cosine_similarity(img_features[i], img_features[i+1])
        avg /= len(img_features)
        result_list.append(avg.cpu().detach().numpy())
    return np.mean(np.array(result_list))


if __name__ == "__main__":
    fc = get_frame_consistency(args.vpath)
    
    print(f'frame consistency : {fc}')
    