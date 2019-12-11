import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from utils import TestDatasetFromFolder, display_transform
from model import Generator


# global parameters
UPSCALE_FACTOR = 4
MODEL_NAME = "SR4_selected/"
BENCHMARK_NAME = "netG_epoch_40.pth"
TRAIN_HR_DIR = "../dataset_1/train_images_hr_selected/"
TRAIN_LR_DIR = "../dataset_1/train_images_lr_selected/"
VAL_HR_DIR = "../dataset_1/val_images_hr_selected/"
VAL_LR_DIR = "../dataset_1/val_images_lr_selected/"
# generated path
TRAIN_SR_DIR = "../dataset_1/train_images_sr_selected/"
VAL_SR_DIR = "../dataset_1/val_images_sr_selected/"
TRAIN_VAL = "both"

RUNCASE = 1
if RUNCASE == 1:
    VAL_HR_DIR = "../dataset_1/val_images_hr_selected2/"
    VAL_LR_DIR = "../dataset_1/val_images_lr_selected2/"
    VAL_SR_DIR = "../dataset_1/val_images_sr_selected2/"
    TRAIN_VAL = "val"

RUNCASE = 2
if RUNCASE == 2:
    VAL_HR_DIR = "../dataset_1/val_images_hr_selected3/"
    VAL_LR_DIR = "../dataset_1/val_images_lr_selected3/"
    VAL_SR_DIR = "../dataset_1/val_images_sr_selected3/"
    TRAIN_VAL = "val"


out_path_train = TRAIN_SR_DIR
out_path_val = VAL_SR_DIR
if not os.path.exists(out_path_train):
    os.makedirs(out_path_train)
if not os.path.exists(out_path_val):
    os.makedirs(out_path_val)

model_path = "results/" + MODEL_NAME + "net_weights/" + BENCHMARK_NAME
model = Generator(scale_factor=UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(model_path))


test_set_train = TestDatasetFromFolder(TRAIN_HR_DIR, TRAIN_LR_DIR)
test_loader_train = DataLoader(dataset=test_set_train, num_workers=0, batch_size=1, shuffle=False)
test_bar_train = tqdm(test_loader_train, desc='[testing train datasets]')
test_set_val = TestDatasetFromFolder(VAL_HR_DIR, VAL_LR_DIR)
test_loader_val = DataLoader(dataset=test_set_val, num_workers=0, batch_size=1, shuffle=False)
test_bar_val = tqdm(test_loader_val, desc='[testing val datasets]')

if TRAIN_VAL == "both" or TRAIN_VAL == "train":
    results_train = {'psnr': [], 'ssim': []}
    with torch.no_grad():
        for lr_image, hr_image, image_names in test_bar_train:
            image_name = image_names[0]
            if torch.cuda.is_available():
                lr_image = lr_image.cuda()
                hr_image = hr_image.cuda()

            sr_image = model(lr_image)
            mse = ((hr_image - sr_image) ** 2).data.mean()
            psnr = 10 * log10(1 / mse)
            ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

            utils.save_image(sr_image.data.cpu().squeeze(0), out_path_train + image_name, padding=0)

            results_train['psnr'].append(psnr)
            results_train['ssim'].append(ssim)
    results_train_psnr = sum(results_train['psnr']) / len(results_train['psnr'])
    results_train_ssim = sum(results_train['ssim']) / len(results_train['ssim'])
    print("train psnr: %f" % results_train_psnr)
    print("train ssim: %f" % results_train_ssim)


if TRAIN_VAL == "both" or TRAIN_VAL == "val":
    results_val = {'psnr': [], 'ssim': []}
    with torch.no_grad():
        for lr_image, hr_image, image_names in test_bar_val:
            image_name = image_names[0]
            if torch.cuda.is_available():
                lr_image = lr_image.cuda()
                hr_image = hr_image.cuda()

            sr_image = model(lr_image)
            mse = ((hr_image - sr_image) ** 2).data.mean()
            psnr = 10 * log10(1 / mse)
            ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

            utils.save_image(sr_image.data.cpu().squeeze(0), out_path_val + image_name, padding=0)

            results_val['psnr'].append(psnr)
            results_val['ssim'].append(ssim)
    results_val_psnr = sum(results_val['psnr']) / len(results_val['psnr'])
    results_val_ssim = sum(results_val['ssim']) / len(results_val['ssim'])
    print("val psnr: %f" % results_val_psnr)
    print("val ssim: %f" % results_val_ssim)