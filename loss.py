import torch
from torch import nn
from torchvision.models.vgg import vgg16, vgg19


class SRGAN_Loss(nn.Module):
    '''Loss function for SRGAN, and the paper version is set as the default parameters.'''
    def __init__(self, content_loss="vgg", adversarial_loss="bce", tv_loss_on=False):
        super(SRGAN_Loss, self).__init__()
        assert content_loss in ["vgg", "mse", "both"], "invalid input."
        assert adversarial_loss in ["bce", "L1"], "invalid input."
        assert type(tv_loss_on) == bool, "invalid input."
        vgg = vgg19(pretrained=True)
        network_loss = nn.Sequential(*list(vgg.features)[:36]).eval()  # VGG_54
        for param in network_loss.parameters():
            param.requires_grad = False  # freezing weights
        self.content_loss = content_loss
        self.adversarial_loss = adversarial_loss
        self.tv_loss_on = tv_loss_on
        self.network_loss = network_loss
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        batch_size = out_labels.size(0)
        ### Content Loss
        if self.content_loss == "vgg":
            content_loss = 0.006 * self.mse_loss(self.network_loss(out_images), 
                                                 self.network_loss(target_images))
        elif self.content_loss == "mse":
            content_loss = self.mse_loss(out_images, target_images)
        elif self.content_loss == "both":
            content_loss = 0.006 * self.mse_loss(self.network_loss(out_images), 
                                                 self.network_loss(target_images))
            content_loss += self.mse_loss(out_images, target_images)
        ### Adversarial Loss
        if self.adversarial_loss == "bce":
            adversarial_loss = self.bce_loss(out_labels, torch.ones_like(out_labels))
        elif self.adversarial_loss == "L1":
            adversarial_loss = torch.mean(1 - out_labels)
        ### TV Loss
        if self.tv_loss_on:
            tv_loss = self.tv_loss(out_images)
        else:
            tv_loss = 0
        
        ### Total Loss
        total_loss = content_loss + 0.001 * adversarial_loss + 2e-8 * tv_loss
        
        return total_loss


class TVLoss(nn.Module):
    '''Total Variational Loss.'''
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]