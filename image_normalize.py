# To normalize the image data into 32x32 gray-scale images
# Will be saved as ./data/train.pickle and ~/test.pickle

import numpy as np
import cv2
import pickle
import os


def white_corner(image, h, w):
    vs = h // 4
    hs = w // 4
    return np.sum(image[:vs, :hs] == 255) + np.sum(image[-vs:, -hs:] == 255) \
           + np.sum(image[:vs, -hs:] == 255) + np.sum(image[-vs:, :hs] == 255) > h * w // 8


def handle(data_type):
    data = {"image": [], "label": []}
    for i in range(10):
        for file_name in os.listdir(f"./data/{data_type}/{i}/"):
            # read as gray scale
            img = cv2.imread(f"./data/{data_type}/{i}/{file_name}", 0)
            h, w = np.shape(img)[:2]
            # inverse if black digit
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            if white_corner(img, h, w):
                img = 255 - img
            # padding to square
            border = max(h, w)
            vertical_padding = (border - h) // 2
            horizon_padding = (border - w) // 2
            padded_img = cv2.copyMakeBorder(img, vertical_padding, vertical_padding,
                                        horizon_padding, horizon_padding,
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # resize to 32x32
            norm_img = cv2.resize(padded_img, (32, 32), interpolation=3)

            data["image"].append(norm_img)
            data["label"].append(i)
        print(f"Label {i} Handled.")
    # save as pickle
    with open(f"./data/{data_type}_data.pickle", "wb") as f:
        pickle.dump(data, f)


handle("train")
handle("test")
