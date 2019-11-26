import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class TrainDatasetFromFolder(Dataset):
    def __init__(self, hr_dir, lr_dir):
        super(TrainDatasetFromFolder, self).__init__()
        assert os.listdir(hr_dir) == os.listdir(lr_dir), "HR and LR images are not 1-to-1"
        self.hr_names = [os.path.join(hr_dir, x) for x in os.listdir(hr_dir)]
        self.lr_names = [os.path.join(lr_dir, x) for x in os.listdir(lr_dir)]

    def __getitem__(self, index):
        hr_image = transforms.functional.to_tensor(Image.open(self.hr_names[index]))
        lr_image = transforms.functional.to_tensor(Image.open(self.lr_names[index]))
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_names)


class ValDatasetFromFolder(Dataset):
    def __init__(self, hr_dir, lr_dir):
        super(ValDatasetFromFolder, self).__init__()
        assert os.listdir(hr_dir) == os.listdir(lr_dir), "HR and LR images are not 1-to-1"
        self.hr_names = [os.path.join(hr_dir, x) for x in os.listdir(hr_dir)]
        self.lr_names = [os.path.join(lr_dir, x) for x in os.listdir(lr_dir)]
        self.img_names = os.listdir(hr_dir)

    def __getitem__(self, index):
        hr_image = transforms.functional.to_tensor(Image.open(self.hr_names[index]))
        lr_image = transforms.functional.to_tensor(Image.open(self.lr_names[index]))
        return lr_image, hr_image, self.img_names[index]

    def __len__(self):
        return len(self.hr_names)
