# ---------------------------------------------
# 2019 DLP final project
# implementation of Pairwise Body-Part Attention for Recognizing Human-Object Interactions
# Written by Chihchia Li, alter027
# ---------------------------------------------

import torch
import torch.nn as nn
import cv2
from torchvision import transforms
import random

import upper_module as um
import vgg as vg
import config as c

upper = um.upper_module(30)
trans = transforms.Compose([
    transforms.ToTensor(),
])
persons = [[0, 0, 1000, 1000]]
img_path = '/home/pj19/DL_final_project/input_img/'
img_name = '000000035005.jpg'
vgg = vg.vgg16()

img = cv2.imread(img_path+img_name)
print('img',img)
upper_res = upper(img, persons, 1, img_name) # for a single person
print(upper_res)