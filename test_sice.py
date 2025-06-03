import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.archs.ecmambaincontext_arch import ECMambaIncontext
from skimage import img_as_ubyte

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='Image Enhancement using ECMamba')

parser.add_argument('--input_dir', default='dataset/SICE/test/Input', type=str, help='Directory of test input images')
parser.add_argument('--gt_dir', default='dataset/SICE/test/GT', type=str, help='Directory of gt images')
parser.add_argument('--result_dir', default='test-results', type=str, help='Directory for results')
parser.add_argument('--weights', default='weights/SICE_weight.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='SICE', type=str, help='Test Dataset')

args = parser.parse_args()


####### Load yaml #######
yaml_file = 'options/ecmamba_test.yml'
weights = args.weights

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = ECMambaIncontext(**x['network_g'])

total = sum([param.nelement() for param in model_restoration.parameters()])
print('total parameters:', total)

checkpoint = torch.load(weights)
checkpoint_name = os.path.basename(weights).split('.')[0]
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",weights)
model_restoration.cuda()
model_restoration.eval()


factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, args.dataset, checkpoint_name)
os.makedirs(result_dir, exist_ok=True)


input_paths = natsorted(glob(os.path.join(args.input_dir, '*.png')) + glob(os.path.join(args.input_dir, '*.JPG')))
gt_paths = natsorted(glob(os.path.join(args.gt_dir, '*.png')) + glob(os.path.join(args.gt_dir, '*.JPG')))

psnr = []
ssim = []

psnr_high = []
ssim_high = []

psnr_low = []
ssim_low = []

with torch.inference_mode():
    for inp_path, gt_path, idx in tqdm(zip(input_paths, gt_paths, range(len(input_paths))), total=len(gt_paths)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]

        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        if h * w >= 2800000:
            input_1 = input_[:, :, :, 1::2]
            input_2 = input_[:, :, :, 0::2]

            restored_1 = model_restoration(input_1)[0]
            restored_2 = model_restoration(input_2)[0]
            restored = torch.zeros_like(input_)
            restored[:, :, :, 1::2] = restored_1
            restored[:, :, :, 0::2] = restored_2
        else:
            restored = model_restoration(input_)[0]

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]
        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        target = np.float32(utils.load_img(gt_path)) / 255
        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0]+'.png')), img_as_ubyte(restored))

        psnr_ = utils.PSNR(target, restored)
        ssim_ = utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored))

        if idx < 30:
            psnr_high.append(psnr_)
            ssim_high.append(ssim_)
        else:
            psnr_low.append(psnr_)
            ssim_low.append(ssim_)

        psnr.append(psnr_)
        ssim.append(ssim_)
        
psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))

psnr_high = np.mean(np.array(psnr_high))
ssim_high = np.mean(np.array(ssim_high))
print("PSNR_high: %f " % (psnr_high))
print("SSIM_high: %f " % (ssim_high))

psnr_low = np.mean(np.array(psnr_low))
ssim_low = np.mean(np.array(ssim_low))
print("PSNR_low: %f " % (psnr_low))
print("SSIM_low: %f " % (ssim_low))

