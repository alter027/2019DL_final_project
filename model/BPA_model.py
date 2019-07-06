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

import faster_rcnn.faster_rcnn as fr
import model.lower_module as lm
import model.upper_module as um
import model.vgg as vgg
import config as c

class BPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.faster = fr.FasterRCNN()
        self.lower_module = lm.lower_module()
        self.upper_module = um.upper_module()
        self.vgg = vgg.vgg16()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.predict = nn.Linear(c.Concat_vec_len*2, c.HOIlen)
    
    def forward(self, imgs_path):
        # get all bounding box of all images
        bboxs = self.faster(imgs_path)

        # process every images one by one
        for obj in bboxs:
            img_path = obj[0]
            bbox = obj[1]

            # sample 3 human and 4 object proposals
            ## Notice! if the number of human and object is insufficient
            ## we should pad zero in future process
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

            # read image and get it's feature map
            img = cv2.imread(img_path)
            img = transform(img).unsqueeze_(0)
            feature_map = self.vgg(img)

            # lower line
            ## dim 0: person
            ## dim 1: hidden vector
            lower_res = self.lower_module(feature_map, persons, objects,\
                 feature_map.shape[-1]/img.shape[-1])
            
            # upper line
            upper_res = self.upper_module(feature_map, )

            # Concat and MIL
            hidden_vec = torch.cat((lower_res, upper_res), 1)
            predict = self.predict(hidden_vec)

            res, _ = torch.max(predict, 0)

            return res
