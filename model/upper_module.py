import json
import torch
import torch.nn as nn
import numpy as np
import heapq
from heapq import heappush, heappop

import model.config as c

class upper_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.fx = c.Body_feature_x
        self.fy = c.Body_feature_y
        self.RoI = nn.AdaptiveMaxPool2d((self.fx, self.fy))
        self.attention = nn.Linear(self.fx*self.fy, 1)
        self.fc6 = nn.Linear(self.fx*self.fy, c.Concat_vec_len)

    def forward(self, features, ratio, image_name, mode='train', k=10):
        # out: tensor -> [person, C(10, 2), feature map,]
        # res_global = []
        res_global = torch.tensor([])
        feature_pairs = self.get_body_pair(features, ratio, image_name, mode)
        
        heap, RoI_pairs = [], []
        # RoI pooling & build heap
        for ind, pair in enumerate(feature_pairs):
            pair = self.RoI(pair).view(-1)
            RoI_pairs.append(pair)
            attscore = self.attention(pair)
            # since this is a minheap, therefore we have a negative sign
            heappush(heap, [-attscore, ind])
        top_k_ind = {i[1]:i[0] for i in heapq.nsmallest(k, heap)}
        for ind, pair in enumerate(RoI_pairs):
            if ind in top_k_ind:
                RoI_pairs[ind] *= -top_k_ind[ind]
            else:
                RoI_pairs[ind] *= 0
        # res_global.append(RoI_pairs)
        RoI_pairs = torch.tensor(RoI_pairs).unsqueeze_(0)
        res_global = torch.cat((res_global, RoI_pairs))

        # output layer
        upper_res = self.fc6(res_global)
        return upper_res

    # def test(self, person, bbox): # output: whether the person is chosen
    #     # see the json format in https://github.com/MVIG-SJTU/AlphaPose/blob/master/doc/output.md
    #     # count how may parts are in the bbox
    #     cnt = 0
    #     print(len(person['joints']))
    #     for i in range(18):
    #         x = person['joints'][i*3]
    #         y = person['joints'][i*3+1]
    #         # not sure the format of bbox
    #         if bbox[1]<=x and x<=bbox[3] and bbox[0]<=y and y<=bbox[2]: 
    #             cnt += 1
    #     if cnt >= 16: # most of body parts are in bbox would be fine
    #         return True

    def cut(self, person): # return 10 body part bbox
        # targets: we only choose some specific parts
        targets, res = [1, 3, 4, 6, 7, 9, 10, 12, 13], []
        joints = person['joints']
        size = abs(joints[2*3]-joints[5*3])/2 # use distance from LShoulder to RShoulder as size of bbox
        for i in targets:
            x = joints[i*3]
            y = joints[i*3+1]
            res.append([int(np.round((x-size))), int(np.round((y-size))),\
                 int(np.round((x+size))), int(np.round((y+size)))])
        # Pelvis -> LHip to RHip (where Pelvis is not detect in this datset)
        x = (joints[8*3]+joints[11*3])/2
        y = (joints[8*3+1]+joints[11*3+1])/2
        res.append([int(np.round((x-size))), int(np.round((y-size))),\
            int(np.round((x+size))), int(np.round((y+size)))])
        return res

    def get_body_part(self, image_name, mode): 
        # ouput: body part bboxes
        name = image_name.split('.')[0]
        filename = f'/home/pj19/twoTFolder/DL_final_project/data/{mode}_alphapose/sep-json/{name}.json'
        # filename = '/home/pj19/twoTFolder/DL_final_project/AlphaPose/examples/res/alphapose-results.json'
        with open(filename) as json_file:
            #  load alphapose data from json file
            pic = json.load(json_file)
            people = pic['bodies']
            res = []
            for i, person in enumerate(people): 
                res.append(self.cut(person))
            return res

    def get_body_pair(self, feature, ratio, image_name, mode): # output: feature map pairs
        parts = self.get_body_part(image_name, mode)
        res = []
        # C(k=10, 2) kinds of union region, and get the correspond feature
        for i in range(10):
            for j in range(10):
                if i==j:
                    continue
                