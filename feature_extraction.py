# To extract HOG feature from image data
# Will be saved under ./data/
import cv2
import pickle
from tqdm import tqdm
import numpy as np

winSize = (8, 8)
blockSize = (4, 4)
blockStride = (2, 2)
cellSize = (2, 2)
n_bins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, n_bins)

winStride = (6, 6)
padding = (6, 6)


def handle(data_type):
    with open(f"./data/{data_type}_data.pickle", "rb") as f:
        data = pickle.load(f)
    hog_data = {"hog": [], "label": data["label"]}
    for img in tqdm(data["image"], ncols=80):
        img_hog = hog.compute(img, winStride, padding).reshape((-1,))
        hog_data["hog"].append(img_hog)
    with open(f"./data/{data_type}_hog.pickle", "wb") as f:
        pickle.dump(hog_data, f)

handle("train")
# handle("test")
