import os
from math import log10
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
import pytorch_ssim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import Generator, Discriminator
from loss import SRGAN_Loss
from utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, scale_lr2hr


### global parameters
RANDOM_SEED = 1
SCALE_FACTOR = 4
TRAIN_HR_DIR = "../dataset_1/train_images_hr_15k/"
TRAIN_LR_DIR = "../dataset_1/train_images_lr_15k/"
VAL_HR_DIR = "../dataset_1/val_images_hr_15k/"
VAL_LR_DIR = "../dataset_1/val_images_lr_15k/"
RESULTS_DIR = "results/" + "SR" + str(SCALE_FACTOR) + "_15k" + "/"
# network parameters
CONTENT_LOSS = "both"  # try "both" in future test
ADVERSARIAL_LOSS = "L1"
TV_LOSS_ON = True
# training parameters
BATCH_SIZE = 1
NUM_EPOCHS = 10
NUM_WORKERS = 0  # workers for loading data
# validating parameters
VAL_IMAGE_RAND_NUM = 10
VAL_IMAGE_INDEX = ["96_100.tif", "543_120.tif", "367_116.tif", "1038_81.tif", "1116_47.tif", "1402_56.tif", "2351_5.tif", "1402_12.tif", "205_0.tif", "359_25.tif"]
### overwrite default global settings
# RUN_CASE = None  # default
# RUN_CASE = 0  # ~500 sample images
# RUN_CASE = 1  # 5000 images
# RUN_CASE = 2  # 1000 512 * 512 images
RUN_CASE = 3  # 6000 selected images


if RUN_CASE == 0:
    NUM_EPOCHS = 50
    TRAIN_HR_DIR = "temp_data/train_images_hr/"
    TRAIN_LR_DIR = "temp_data/train_images_lr/"
    VAL_HR_DIR = "temp_data/val_images_hr/"
    VAL_LR_DIR = "temp_data/val_images_lr/"
    RESULTS_DIR = "results/" + "SR" + str(SCALE_FACTOR) + "/"
    VAL_IMAGE_RAND_NUM = 5
elif RUN_CASE == 1:
    TRAIN_HR_DIR = "../dataset_1/train_images_hr_5k/"
    TRAIN_LR_DIR = "../dataset_1/train_images_lr_5k/"
    VAL_HR_DIR = "../dataset_1/val_images_hr_5k/"
    VAL_LR_DIR = "../dataset_1/val_images_lr_5k/"
    RESULTS_DIR = "results/" + "SR" + str(SCALE_FACTOR) + "_5k" + "/"
    VAL_IMAGE_RAND_NUM = 20
    VAL_IMAGE_INDEX = []
elif RUN_CASE == 2:
    TRAIN_HR_DIR = "../dataset_1/train_images_hr_512_1k/"
    TRAIN_LR_DIR = "../dataset_1/train_images_lr_512_1k/"
    VAL_HR_DIR = "../dataset_1/val_images_hr_512_1k/"
    VAL_LR_DIR = "../dataset_1/val_images_lr_512_1k/"
    RESULTS_DIR = "results/" + "SR" + str(SCALE_FACTOR) + "_512_1k" + "/"
    VAL_IMAGE_RAND_NUM = 20
    VAL_IMAGE_INDEX = []
elif RUN_CASE == 3:
    NUM_EPOCHS = 50
    TRAIN_HR_DIR = "../dataset_1/train_images_hr_selected/"
    TRAIN_LR_DIR = "../dataset_1/train_images_lr_selected/"
    VAL_HR_DIR = "../dataset_1/val_images_hr_selected/"
    VAL_LR_DIR = "../dataset_1/val_images_lr_selected/"
    RESULTS_DIR = "results/" + "SR" + str(SCALE_FACTOR) + "_selected" + "/"
    VAL_IMAGE_RAND_NUM = 20
    VAL_IMAGE_INDEX = []



torch.manual_seed(RANDOM_SEED)


train_dataset = TrainDatasetFromFolder(hr_dir=TRAIN_HR_DIR, lr_dir=TRAIN_LR_DIR)
train_loader = DataLoader(dataset=train_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = ValDatasetFromFolder(hr_dir=VAL_HR_DIR, lr_dir=VAL_LR_DIR)
val_loader = DataLoader(dataset=val_dataset, num_workers=NUM_WORKERS, batch_size=1, shuffle=False)

netG = Generator(scale_factor=SCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator(dense_choice=1)
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))


generator_criterion = SRGAN_Loss(content_loss=CONTENT_LOSS, 
                                 adversarial_loss=ADVERSARIAL_LOSS, 
                                 tv_loss_on=TV_LOSS_ON)
bceloss = torch.nn.BCELoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()
    
optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())
# optimizerG = optim.SGD(netG.parameters(), lr=0.005, momentum=0.9)
# optimizerD = optim.SGD(netD.parameters(), lr=0.005, momentum=0.9)




results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
out_path_val = RESULTS_DIR + "val_predict/"
if not os.path.isdir(out_path_val):
    os.mkdir(out_path_val)
out_path_net = RESULTS_DIR + "net_weights/"
if not os.path.isdir(out_path_net):
    os.mkdir(out_path_net)
val_image_rand = np.random.choice(np.arange(len(val_dataset)), size=VAL_IMAGE_RAND_NUM, replace=False)

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    
    ### training
    for lr_img, hr_img in train_bar:
        g_update_first = True
        batch_size = lr_img.size(0)
        running_results['batch_sizes'] += batch_size
        
        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(hr_img)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(lr_img)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img)
        fake_out = netD(fake_img)
        # d_loss = -fake_out.log().sum()  # paper d loss ?
        d_loss = 1 - real_out.mean() + fake_out.mean()  # L1 loss
        # d_loss = bceloss(real_out, torch.ones_like(real_out)) + bceloss(fake_out, torch.zeros_like(fake_out))
        d_loss.backward(retain_graph=True)
        optimizerD.step()
        
        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()

        fake_img = netG(z)
        fake_out = netD(fake_img).mean()

        optimizerG.step()

        # loss for current batch before optimization 
        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.sum().item()
        running_results['g_score'] += fake_out.sum().item()

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(hr): %.4f D(G(lr)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))
        pass
    
    ### evaluating
    netG.eval()
    out_path_val_epoch = out_path_val + "epoch_%d/" % epoch
    if not os.path.isdir(out_path_val_epoch):
        os.mkdir(out_path_val_epoch)
    with torch.no_grad():
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_bar = tqdm(val_loader)
        val_images = []
        val_names = []
        for i, (val_lr, val_hr, val_name) in enumerate(val_bar):
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size
            lr = val_lr
            hr = val_hr
            lr2hr = scale_lr2hr((256, 256))(lr.squeeze(0))
            # lr2hr = scale_lr2hr((512, 512))(lr.squeeze(0))
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            sr = netG(lr)

            batch_mse = ((sr - hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))
            if i in val_image_rand or val_name[0] in VAL_IMAGE_INDEX:
	            val_images.extend(
	                [display_transform()(lr2hr.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
	                 display_transform()(sr.data.cpu().squeeze(0))])
	            val_names.extend(val_name)
        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 3)
        # val_save_bar = zip(tqdm(val_images, desc='[saving validating results]'), val_names)
        for image, name in zip(val_images, val_names):
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, out_path_val_epoch + '%s.png' % name.strip(".tif"), padding=5)
    
    ### save model parameters
    torch.save(netG.state_dict(), out_path_net + "netG_epoch_%d.pth" % epoch)
    torch.save(netD.state_dict(), out_path_net + "netD_epoch_%d.pth" % epoch)
    
    ### save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    if epoch % 1 == 0 and epoch != 0:
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(1, epoch + 1))
        data_frame.to_csv(RESULTS_DIR + 'train_stats.csv', index_label='Epoch')