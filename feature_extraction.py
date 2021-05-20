# To extract HOG feature from image data
# Will be saved as ./data/train_hog.pickle and ~/test_hog.pickle
import os
import cv2
import pickle
from tqdm import tqdm
import numpy as np

def img_show(imag, name="img", key=0):
    cv2.namedWindow(name)
    h, w = np.shape(imag)[:2]
    # h *= (300 / w)
    # h = int(h)
    # cv2.resizeWindow(name, w, h)
    cv2.imshow(name, imag)
    cv2.waitKey(key)
    cv2.destroyAllWindows()

winSize = (16, 16)
blockSize = (8, 8)
blockStride = (4, 4)
cellSize = (4, 4)
n_bins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, n_bins)

winStride = (10, 10)
padding = (4, 4)


def handle(data_type):
    data = {"hog": [], "label": []}
    for i in range(10):
        for file_name in os.listdir(f"./data/{data_type}/{i}/"):
            # read as original image
            img = cv2.imread(f"./data/{data_type}/{i}/{file_name}")
            norm_img = cv2.resize(img, (32, 32))
            hog_f = hog.compute(norm_img, winStride, padding).reshape((-1,))
            data["hog"].append(hog_f)
            data["label"].append(i)
            # print(len(hog_f))
        print(f"Label {i} Handled.")
    # save as pickle
    with open(f"./data/{data_type}_hog.pickle", "wb") as f:
        pickle.dump(data, f)


handle("train")
handle("test")
