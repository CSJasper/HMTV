# Pytorch Implementation of paper "Human Motion Aware Text-to-Video Generation with Explicit Camera Control"

[Project Page](https://anonymous.4open.science/r/HMTV_docs-5A6C)

## Installation

### Environment
Our code was tested on CUDA 11.7, python3.8.
You can run this code with a single RTX 3090-24G

Using `environment.yml`,
```
conda env create -f environment.yml
conda active hmtv
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
Before running demo you should run the script below
```
bash prepare_t2v.sh
```

This script clone T2V-Zero and prepare the environment.

If you want to try another T2V models run `clean_t2v.sh` before cloning them.