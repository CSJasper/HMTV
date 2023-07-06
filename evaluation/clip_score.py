import torch
from torchmetrics.multimodal import CLIPScore
from torchmetrics.functional.multimodal import clip_score
import cv2
import os.path as osp
import argparse
import os
from tqdm import tqdm
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--vpath", type=str)
parser.add_argument("--openpose", type=bool,default=False)

args = parser.parse_args()

def get_clip_score(vpath: str):
    video_path = vpath
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()

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
        if args.openpose==False:
            list_score=[]
            for frm in frames:
                prompt=video_name[:-4].replace("_"," ")
                img=torch.from_numpy(np.array(frm).astype("float32")).cuda()
                score = metric(img, prompt)
                list_score.append(score.cpu().detach().numpy())
            result_list.append(np.mean(np.array(list_score)))
        else:
            list_score=[]
            prompt=""
            name_list_video=video_name.split("_")
            length_video_name=len(name_list_video)
            for num_lst_n, vid_name in enumerate(name_list_video):
                if num_lst_n==length_video_name-1:
                    continue
                elif num_lst_n==length_video_name-2:
                    prompt+=vid_name
                else:
                    prompt+=vid_name+" "
            for frm in frames:
                img=torch.from_numpy(np.array(frm).astype("float32")).cuda()
                score = metric(img, prompt)
                list_score.append(score.cpu().detach().numpy())
            result_list.append(np.mean(np.array(list_score)))

    result = np.mean(np.array(result_list))
    return result
    
    


if __name__ == "__main__":
    clip_score = get_clip_score(args.vpath)
    print(f"clip score : {clip_score}")