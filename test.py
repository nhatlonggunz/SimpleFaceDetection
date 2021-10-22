import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mp_img
import cv2
from  tensorflow.examples.tutorials.mnist import input_data

img = cv2.imread('lan (1).jpg')
img = img / 255
def convert_to_grayscale(img):
    return np.dot(img[..., : 3], [0.299, 0.587, 0.114])
img = convert_to_grayscale(img)
print(img)
print(img.shape)
cv2.imshow('img', img)
cv2.waitKey(0)