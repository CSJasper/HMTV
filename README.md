# Pytorch Implementation of paper "Human Motion Aware Text-to-Video Generation with Explicit Camera Control" (WACV 2024)


Our Project Page would be available soon.


![Our frameworks](assets/main_figure_final.jpg)

## Installation

### Environment
Our code was tested on CUDA 11.7, python3.8.
You can run this code with a single RTX 3090-24G

Using `environment.yml`,
```
conda env create -f environment.yml
conda activate hmtv
```
or you can maunally create conda environment and use `requirements.txt` as below

```
conda create -n hmtv python=3.8 -y
conda activate hmtv
pip install -r requirements.txt
```

If you have trouble with `clip`, you can install with these scripts below

```
pip install ftfy regex tqdm

pip install git+https://github.com/openai/CLIP.git 
```


## Demo
Demo is available in `demo.ipynb`
Before running demo you should run the script below.
```
bash prepare_t2v.sh  
```

This script provides T2M model and one of the T2V model in the paper.

After that you should download pretrained models, checkpoints and other things to run inference code for T2M model.

Specifically, you need to run

```
bash text2motion/dataset/prepare/download_model.sh
nash text2motion/dataset/prepare/download_smpl.sh
```
and npy files of mean and std.(`mean.npy`, `std.npy`)

The details are provided in this [link](https://github.com/Mael-zys/T2M-GPT).

If you want to try another T2V models run `clean_t2v.sh` before cloning them.


## Solutions for error related to environment setting

### Problems related to argument parser

We have discovered that there is an issue with the argument parser in certain environments. If you are experiencing the same problem, try changing the code as follows:

Modify the code
```
    return parser.parse_args()
```

to 

```
    return parser.parse_args(args=[])
```
 in `text2motion/options/option_transformer.py`


### Problems related to paths
We provided demo code in our repo. However, in some case there might be differnet setting in working directory paths. So, if this is the case you should change some parts of the code that we provided.

In second code block in `demo.ipynb`, it would help if you change this code below

```
from text2motion.text2motion import predict
```

to

```
from text2motion import predict
```

Additionally, you must enter the absolute paths for mean and std, not the relative paths. This can be changed in the following section of text2motion/text2motion.py:"

```
mean = torch.from_numpy(np.load("YOUR ABSOLUTE PATH HERE"))
std = torch.from_numpy(np.load(YOUR ABSOLUTE PATH HERE))
```

