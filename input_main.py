import numpy as np
from scipy import ndimage
import os
import cv2
import errno
import sys
import logging
import shutil
import matplotlib.pyplot as plt


def create_list_data(path):
    BOSS = os.listdir('./' + path)
    save = []

    cnt = 0

    lab_name = []
    lab_ind = dict()

    for lab in BOSS:
        
        lab_name.append(lab)
        lab_ind[lab] = cnt
        cnt = cnt + 1

        for img in os.listdir(path + '/' + lab):
            if img.endswith('jpg'):
                save.append(path + '/' + lab + '/' + img)

    return save, lab_name, lab_ind

def random_list_data(data):
    data = np.random.permutation(data)
    return data

def normalize(img):
    img = np.float32(img)
    img = img / 255
    return img

def convert_to_grayscale(img):
    return np.dot(img[..., : 3], [0.299, 0.587, 0.114])

def one_hot(x, num_lab):
    nx = num_lab + 1
    return np.eye(nx)[x]

def read_batch(data, lab_ind, l, r):
    img_list = []
    lab_list = []

    l = int(l)
    r = int(r)

    for i in range(l, r):
        lab = data[i].split('/')[-2]
        
        lab_list.append(lab_ind[lab])

        img = cv2.imread(data[i])
        img_list.append(img)

    img_list = normalize(img_list)
    img_list = convert_to_grayscale(img_list)

    MAX = max(lab_ind.values())
    lab_list = one_hot(lab_list, MAX)

    return img_list, lab_list

# gg, lab_name, lab_ind = create_list_data('./datasets')
# gg = random_list_data(gg)

# img, lab = read_batch(gg, lab_ind, 0, 2)


# print(img)
# print(lab)
# print(img.shape)

# cv2.imshow('img', img[0])
# cv2.waitKey(0)