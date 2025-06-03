#  [NeurIPS 2024] ECMamba: Consolidating Selective State Space Model with Retinex Guidance for Efficient Multiple Exposure Correction

<h4 align="center">Wei Dong<sup>1,*</sup>, Han Zhou<sup>1,*</sup>, Yulun Zhang<sup>2</sup>, Xiaohong Liu<sup>2,&dagger;</sup>, Jun Chen<sup>1</sup></center>
<h4 align="center"><sup>1</sup>McMaster University, <sup>2</sup>Shanghai Jiao Tong University, 
<h4 align="center"><sup>*</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding Author</center></center>
  
### Introduction
This repository represents the official implementation of our NeurIPS 2024 paper titled **ECMamba: Consolidating Selective State Space Model with Retinex Guidance for Efficient Multiple Exposure Correction**. If you find this repo useful, please give it a star ⭐ and consider citing our paper in your research. Thank you for your interest. 

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

We present ECMamba, the first mamba-based framework for multiple exposure correction and low-light image enhancement.

- **Mamba-based Multiple Exposure Correction**: exploit mamba-based framework to process images with adverse illumination with high efficiency.
- **Dual-path Retinex-guided Restoration Framework**: develop a dual-path Retinex-guided restoration pipeline by introducing two intermediary spaces based on Retinex theory.
- **Feature-aware Scanning Strategy**: Different from direction-sensitive scanning method, we design a feature-aware 2D selective scanning mechanism to transform 2D iamge or feature maps into 1D sequences. 

## 📢 News
**2025-06-03** This repo has been updated. Pre-trained weights and test codes are released!

### Overall Framework
![teaser](images/framework.png)


## 🛠️ Setup

The inference code was tested on:

- Python 3.9, CUDA 11.7, PyTorch 2.0.1 + cu117.

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/LowLevelAI/ECMamba.git
cd ECMamba
```

### 💻 Dependencies

- **Make Conda Environment: Using [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to create the environment:** 

    ```bash
    conda create -n ecmamba python=3.9
    conda activate ecmamba
    ```
- **Then install dependencies:**
  - Install Pytorch

  ```bash
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
  ```
  - Install `mamba_ssm` library
  ```bash
  pip install causal_conv1d==1.0.0
  pip install mamba_ssm==1.0.1
  ```
  - Install `DCNv4` library
  ```bash
  pip install -U openmim
  mim install mmcv-full
  pip install timm mmdet
  pip install opencv-python termcolor yacs pyyaml scipy
  pip install DCNv4
  ```
  - Install other dependencies
  ```bash
  pip install numpy==1.26.4 transformers==4.48.2 opencv-python natsort scikit-image timm matplotlib
  ```

- **Revise `DCNv4` CUDA extensions:**
  In `/anaconda3/envs/ecmamba/lib/python3.9/site-packages/DCNv4/modules/dcnv4.py` line 153, replace `retunr x` with **`retune x, offset_mask`**.



