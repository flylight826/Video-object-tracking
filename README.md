# Introduction

​		本模型基于SAMURAI开源项目进行复现和优化，SAMURAI是华盛顿大学团队提出的基于Segment  Anything Model 2 的视觉目标跟踪增强框架。我们在原工作基础上进行了进一步的算法调优和工程实现， 显著提升了在复杂场景下的跟踪性能。 

​		主要用于参加2025年全国校园人工智能算法精英大赛  赛道：算法挑战赛  赛题：视觉+几何+语义：多源异构数据协同的视频目标跟踪，如想获取模型权重，链接: https://pan.baidu.com/s/1NfRKSIOFhhD6sMqGCNuByw?pwd=amnw 提取码: amnw。



# Getting Started

## SAMURAI Installation 

```
conda create -n sam python==3.11
conda activate sam
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```

Install other requirements:
```
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru

#### Unknown problem,need to install Pillow again

pip uninstall Pillow
pip install Pillow
```



## SAM 2.1 Checkpoint Download

```
# If the model file has been downloaded, please move it to the specified path:
sam2/checkpoints

# Alternatively, use the following command to download it automatically.
cd checkpoints
bash ./download_ckpts.sh
cd ..
```



## Data Preparation

Please prepare the data in the following format:
```
SAMURAI is based on pre-trained SAM2, with motion memory augmentation, 
enabling zero-shot tracking without training.

for test only
data/
├── lasot/
│   ├── 001/
│   │   ├── color
│   │   ├── depth
│   │   ├── groundtruth.txt
│   │   ├── nlp.txt
│   ├── 002/
│   ├── 003/
│   ├── ...

```



## Main Inference

```
/samurai-master/scripts/main_inference.py line 65
video_folder = "#set your /path/to/data/lasot"  

cd /path/to/samurai-master
python scripts/main_inference.py 
```
