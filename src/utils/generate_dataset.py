import cv2 as cv
import os
from glob import glob
import numpy as np

DSIZE = (128, 128)
DATASET_DIR = "../../data/GOPRO/train"
FILTER_PATTERN = "*/sharp/*.png"
DATASET_SIZE = 100
# OUTPUT_DIR = f"../../data/GOPRO_{DSIZE[0]}x{DSIZE[0]}/"
OUTPUT_DIR = f"../../data/GOPRO_{DSIZE[0]}x{DSIZE[0]}_GAUSSIAN_BLUR/"
OUTPUT_DIR = f"../../data/GOPRO_{DSIZE[0]}x{DSIZE[0]}_SALT_AND_PAPER/"

REMOVE_DIR = True

RESIZE = True
SALT_AND_PAPER = True
GAUSSIAN_BLUR = False

images = [y for x in os.walk(DATASET_DIR) for y in glob(os.path.join(x[0], FILTER_PATTERN))][:DATASET_SIZE]

print(os.listdir(DATASET_DIR))
# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv

for idx, im_path in enumerate(images):
    print((idx / len(images)) * 100)
    img = cv.imread(im_path)

    if RESIZE:
        img = cv.resize(img, dsize=DSIZE)

    if GAUSSIAN_BLUR:
        row, col, ch = img.shape
        mean = 0
        var = 200
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)

        print("blurring")
        img = img + gauss

    if SALT_AND_PAPER:
        row, col, ch = img.shape
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        out[coords] = 0

        img = out

    cv.imwrite(OUTPUT_DIR + f"/{idx}.png", img)
