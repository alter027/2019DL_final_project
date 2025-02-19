# ---------------------------------------------
# 2019 DLP final project
# implementation of Pairwise Body-Part Attention for Recognizing Human-Object Interactions
# Written by Chihchia Li, alter027
# ---------------------------------------------

import torch
import torch.nn as nn
import numpy as np

import model.config as c

class lower_module(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.fx = c.Global_feature_x # x length of feature map after RoI Pooling
        self.fy = c.Global_feature_y # y length of feature map after RoI Pooling
        self.RoI = nn.AdaptiveMaxPool2d((self.fx, self.fy))
        self.fc6 = nn.Linear(512*self.fx*self.fy*6, c.Concat_vec_len)
        self.device = device

    def forward(self, feature, person_bboxs, obj_bboxs, ratio):
        scene = self.RoI(feature).view(-1)
        total = []
        human = []
        objects = []

        # iterate persons and objects
        # store the feature map of human, objects and scene
        for i in person_bboxs:
            per_b = []
            for idx, x in enumerate(i[1][:4]):
                per_b.append(int(np.round(x*ratio[idx%2])))
            f = feature[:, :, per_b[1]:per_b[3], per_b[0]:per_b[2]]
            human.append(self.RoI(f).view(-1))

            for j in obj_bboxs:
                obj_b = []
                for idx, x in enumerate(j[1][:4]):
                    obj_b.append(int(np.round(x*ratio[idx%2])))
                f = feature[:, :, min(per_b[1], obj_b[1]):max(per_b[3], obj_b[3]),\
                    min(per_b[0], obj_b[0]):max(per_b[2], obj_b[2])]
                objects.append(self.RoI(f).view(-1))

        shape = human[0].shape
        size = 1
        for i in shape:
            size *= i
        
        for _ in range(3-len(human)):
            human.append(torch.zeros(size).to(self.device))
        for _ in range(12-len(objects)):
            objects.append(torch.zeros(size).to(self.device))
        
        for i in range(3):
            total.append(torch.cat((human[i], objects[i*4], objects[i*4+1]\
                , objects[i*4+2], objects[i*4+3], scene)).unsqueeze_(0))
        res_global = torch.cat((total[0], total[1], total[2]))
        lower_res = self.fc6(res_global)
        return lower_res