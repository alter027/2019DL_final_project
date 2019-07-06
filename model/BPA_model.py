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
import json

import faster_rcnn.faster_rcnn as fr
import model.lower_module as lm
import model.upper_module as um
import model.vgg as vgg
import model.config as c

class BPA(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.faster = fr.FasterRCNN()
        self.lower_module = lm.lower_module(device)
        self.upper_module = um.upper_module()
        self.vgg = vgg.vgg16()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.predict = nn.Linear(c.Concat_vec_len*2, c.HOIlen)

        with open('data/train_bbox.json', 'r') as f:
            self.train_bboxes = json.load(f)
        with open('data/test_bbox.json', 'r') as f:
            self.test_bboxes = json.load(f)
    
    def forward(self, x, img_name, mode, orig_size):
        img_name = img_name[0]
        if mode == 'train':
            bbox = self.train_bboxes[img_name]
        elif mode == 'test':
            bbox = self.test_bboxes[img_name]

        ## sample 3 human and 4 object proposals
        ## if the number of human or object is insufficient 
        ## we will pad zero in future process
        persons, objects = [], []
        for i in bbox:
            if i[0] == 'person':
                persons.append(i)
            elif i[0] != '__background__':
                objects.append(i)

        try:
            persons = random.sample(persons, 3)
        except:
            pass
        try:
            objects = random.sample(objects, 4)
        except:
            pass

        ## get x feature map 
        # x = transform(x).unsqueeze_(0)
        with torch.no_grad():
            feature_map = self.vgg(x)
        ## lower line
        ## dim 0: person
        ## dim 1: hidden vector
        x_ratio = feature_map.shape[3]/int(orig_size[0][0])
        y_ratio = feature_map.shape[2]/int(orig_size[1][0])
        lower_res = self.lower_module(feature_map, persons, objects,\
             (x_ratio, y_ratio))
        
        ## upper line
        print('upper start')
        upper_res = self.upper_module(feature_map, (x_ratio, y_ratio),\
             img_name, mode)
        print('upper finish')
        input()
        ## Concat and MIL
        hidden_vec = torch.cat((lower_res, upper_res), 1)
        predict = self.predict(hidden_vec)

        res, _ = torch.max(predict, 0)

        return res