import cv2
import numpy as np
import pickle
import os

def img_show(imag, name="img", key=0):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # h, w = np.shape(imag)[:2]
    # h *= (300 / w)
    # h = int(h)
    # cv2.resizeWindow(name, 300, h)
    cv2.imshow(name, imag)
    cv2.waitKey(key)
    cv2.destroyAllWindows()

with open("./data/test_hog.pickle", "rb") as f:
    data = pickle.load(f)
#
for img in data["hog"]:
    img_show(img)

