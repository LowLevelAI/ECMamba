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
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]

def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)

parser = argparse.ArgumentParser(description='Image Enhancement using ECMamba')

parser.add_argument('--input_dir', default='dataset/LOLv1/Test/input', type=str, help='Directory of test input images')
parser.add_argument('--gt_dir', default='dataset/LOLv1/Test/target', type=str, help='Directory of gt images')
parser.add_argument('--result_dir', default='test-results', type=str, help='Directory for results')
parser.add_argument('--weights', default='weights/LOLv1_weight.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='LOLv1', type=str, help='Test Dataset')
parser.add_argument('--GT_mean', default=True, type=bool, help='Use GT mean to adjust the output')

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
if not args.GT_mean:
    result_dir  = os.path.join(args.result_dir, args.dataset + '-no-GT-mean')
else:
    result_dir  = os.path.join(args.result_dir, args.dataset)

os.makedirs(result_dir, exist_ok=True)


input_paths = natsorted(glob(os.path.join(args.input_dir, '*.png')) + glob(os.path.join(args.input_dir, '*.JPG')))
gt_paths = natsorted(glob(os.path.join(args.gt_dir, '*.png')) + glob(os.path.join(args.gt_dir, '*.JPG')))

psnr = []
ssim = []

with torch.inference_mode():
    for inp_path, gt_path in tqdm(zip(input_paths, gt_paths ), total=len(gt_paths)):
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

        restored = model_restoration(input_)[0]

        # Unpad images to original dimensions
        restored_tensor = restored[:,:,:h,:w]
        restored = torch.clamp(restored_tensor,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        target = np.float32(utils.load_img(gt_path))/255

        if args.GT_mean:    
            mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0]+'.png')), img_as_ubyte(restored))

        psnr.append(utils.PSNR(target, restored))
        ssim.append(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))
        
        
psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))

print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))




