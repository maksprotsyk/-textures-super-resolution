import os
from typing import List, Optional

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
import wandb

from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

log = utils.get_logger(__name__)

import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


def pad_image(img, desired_shape=(64, 64)):
    """
    Pads image to specific shape
    :param img: np.array
    :param desired_shape: tuple (h, w)
    :return: padded image
    """
    result = np.zeros((*desired_shape, 4))
    result[:img.shape[0], :img.shape[1], :] = img
    return result


@hydra.main(config_path="../../configs/", config_name="config.yaml")
def restore_textures(config: DictConfig):
    path = "/home/bohdan/Documents/UCU/3/AI/textures_super-resolution/data/minecraft_textures_v1"

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    images_names = os.listdir(path)

    res = []

    for img_path in images_names:
        img_path = os.path.join(path, img_path)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) / 255.0

        # padding image if it's smaller than desired shape
        if (image.shape[0] % 128 != 0) or (image.shape[1] % 128 != 0):
            image = pad_image(image, desired_shape=(128 * ((image.shape[0] // image.shape[0]) + 1),
                                                    128 * ((image.shape[0] // image.shape[0]) + 1)))

        res.append(model(image))

    return res

if __name__ == "__main__":
    restore_textures()
