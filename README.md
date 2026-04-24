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
