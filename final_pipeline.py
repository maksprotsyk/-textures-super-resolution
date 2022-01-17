import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations import HorizontalFlip, RandomCrop, CenterCrop, ShiftScaleRotate
from albumentations.core.composition import Compose
import random
from src.utils.utils import pad_image
import torchvision.transforms.functional as F

from src.models.superres_model import SuperResLitModel
from torchmetrics import SSIM, PSNR

from torchvision.transforms import transforms
import torch
from pytorch_lightning import (
    seed_everything,
)

SAVE_RESULTS = False


def upscale_image(image, model):
    print(f"{image.shape=}")

    # tensor split
    split_x = torch.split(image, 64, dim=1)
    split_y = [torch.split(i, 64, dim=2) for i in split_x]
    flat_list = [item for sublist in split_y for item in sublist]

    # for idx, img in enumerate(flat_list):
    #     cv2.imwrite(f"./images/{idx}.png", (img.permute(1, 2, 0).numpy() + 1) / 2 * 255)

    # upscale process
    for img_idx, img in enumerate(flat_list):
        flat_list[img_idx] = model(img[None, :, :, :])[0, :, :, :]

    # get images back
    x_splits = image.shape[-2] // 64
    y_splits = image.shape[-1] // 64

    x_united = []
    for x in range(0, y_splits * x_splits, x_splits):
        x_united.append(torch.cat(flat_list[x: x + x_splits], dim=2))

    res = torch.cat(x_united, dim=1)

    return res


class SuperResDataset(Dataset):
    """
    Dataset for super-resolution training
    """

    def __init__(self, img_dir, mode: str = "train", test_set_size: int = 10):
        """
        :param img_dir: path to dir with images
        :param mode: 'train' or 'test'
        """
        self.mode = mode
        self.img_dir = img_dir

        # print(f"{self.image_shape=}")
        print(f"path {self.img_dir}")
        # transformations

        # train/test split
        self.test_set_size = test_set_size
        images_names = os.listdir(img_dir)
        random.seed(1)
        random.shuffle(images_names)
        if self.mode == 'train':
            self.data = images_names[self.test_set_size:]
        elif self.mode == 'test':
            self.data = images_names[:self.test_set_size]
        else:
            print(f"Invalid mode {self.mode}")

        # normalization
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data[idx])

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) / 255.0

        # padding image if it's smaller than desired shape
        if ((image.shape[0] % 128) != 0) or ((image.shape[1] % 128) != 0):
            image = pad_image(image, desired_shape=(image.shape[0] + (128 - (image.shape[0] % 128)),
                                                    image.shape[1] + (128 - (image.shape[1] % 128))))

        image = self.norm(image)

        print(f"{int(image.shape[-1] / 2)=}")

        downscaled = F.resize(image, size=int(image.shape[-1] / 2))

        return downscaled.type(torch.FloatTensor), image.type(torch.FloatTensor)


if __name__ == "__main__":
    seed_everything(2002, workers=True)

    dataset = SuperResDataset("/home/bohdan/BANet/data/minecraft_textures_v1",
                              test_set_size=50)

    model = SuperResLitModel().load_from_checkpoint("./checkpoints/SwinIR-128-1-epoch_042.ckpt").cuda()

    psnr_metric = PSNR(data_range=255.0)
    ssim_metric = SSIM(data_range=255.0)
    ssim_metrics = []
    psnr_metrics = []

    with torch.no_grad():
        for idx, (downscaled, image) in enumerate(dataset):
            print(f"{idx/len(dataset)*100}")

            up_scaled = upscale_image(downscaled.cuda(), model)

            ssim_value = ssim_metric(preds=(up_scaled.clone().detach().cpu()[None, :, :, :] + 1) / 2 * 255,
                                     target=(image.clone().detach().cpu()[None, :, :, :] + 1) / 2 * 255)
            psnr_value = psnr_metric(preds=(up_scaled.clone().detach().cpu()[None, :, :, :] + 1) / 2 * 255,
                                     target=(image.clone().detach().cpu()[None, :, :, :] + 1) / 2 * 255)
            ssim_metrics.append(ssim_value.item())
            psnr_metrics.append(psnr_value.item())

            cv2.imwrite(f"./images/res_{idx}.png", (up_scaled.cpu().permute(1, 2, 0).numpy() + 1) / 2 * 255)

            if idx / len(dataset) * 100 > 5:
                break

    print("ssim = ", np.array(ssim_metrics).mean())
    print("psnr = ", np.array(psnr_metrics).mean())
