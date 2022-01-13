# 2021-VRDL-HW4-Image-Super-Resolution
HW4 Introduction: Image super resolution

The proposed challenge is an image super resolution, and we have to train the model to reconstruct a high-resolution image from a low-resolution input. There are three parts in my project: 
1.	Split training dataset to train and valid datasets.
2.	The train images are scaled down by 3 times as the low-resolution images.
3.	Train the model to generate the high-resolution image

I implemented the Lightweight Image Super-Resolution with Information Multi-distillation Network (IMDN) to fix this challenge.

### Environment
- Windows
- Python 3.6.12
- Pytorch 1.2.0
- CUDA 10.0

### IMDN
The project is implemented based on IMDN.
- [IMDN](https://github.com/Zheng222/IMDN)

## Reproducing Submission
To reproduct my submission without retrainig, run inference.ipynb on my Google Drive:
- [inference.ipynb](https://colab.research.google.com/drive/1yvmSBZ8Im0fRhRVcR-1jjSPJHzgH1fg6?usp=sharing)

## All steps including data preparation, train phase and test phase
1. [Dataset Preparation](#dataset-preparation)
2. [Download Pretrained model](#download-pretrained-model)
3. [Training](#training)
4. [Testing](#testing)
5. [Reference](#reference)

### Dataset Preparation
- Download the given [datasets](https://drive.google.com/file/d/1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb/view) from Google Drive
- Run command `tt_train_valid.py` to split the data into train and valid folders 
- Generate the low-resolution images by scaling down train images by 3 times
- Run commands as below to convert png file to npy file
```
python scripts/png2npy.py --pathFrom new_dataset/train_tt/HR --pathTo new_dataset/train_tt/HR_decoded
```
```
python scripts/png2npy.py --pathFrom new_dataset/train_tt/LR --pathTo new_dataset/train_tt/LR_decoded
```

The files in the data folder is reorganized as below:
```
./new_dataset
 ├── train
 │     ├── HR - xxx.png  
 │     ├── LR - xxx.png
 │     ├── HR_decoded - xxx.npy
 │     └── LR_decoded - xxx.npy
 └──  valid
       ├── HR - xxx.png  
       └── LR - xxx.png 
```

### Download Pretrained model
- download the [pretrained model](https://drive.google.com/file/d/1Fsx7E8kw6Qbe0re6cyYAEzozhvFtGx6S/view?usp=sharing) from my Google Drive

### Training
- to train models, run following commands.
```
python train_IMDN.py
```
After training, it may generate a folder named "checkpoint_x3", with weight files

### Testing
- reconstruct a high-resolution image from a low-resolution input
```
!python test_IMDN.py --checkpoint checkpoint_x3/model.pth --upscale_factor 3
```
- download the [pretrained model](https://drive.google.com/file/d/1Fsx7E8kw6Qbe0re6cyYAEzozhvFtGx6S/view?usp=sharing) from my Google Drive
```
!python test_IMDN.py --checkpoint checkpoint_x3/model.pth --upscale_factor 3
```

### Reference
- [IMDN-github](https://github.com/Zheng222/IMDN#readme)
- [IMDN-paper](https://arxiv.org/pdf/1909.11856v1.pdf)



