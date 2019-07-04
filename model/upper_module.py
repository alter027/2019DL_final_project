import json
import torch
from heapq import heappush, heappop

import config as c

class upper_module(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fx = hidden_size
        self.fy = hidden_size
        self.RoI = nn.AdaptiveMaxPool2d((self.fx, self.fy))
        self.attention = nn.linear(self.fx*self.fy, 1)
        self.fc6 = nn.Linear(self.fx*self.fy, c.Concat_vec_len)
    def forward(self, features, people, ratio, image_name, k):
        # out: tensor -> [person, C(10, 2), feature map,]
        res_global = []
        for person in people:
            feature_pairs = get_body_pair(features, person, image_name)
            heap, RoI_pairs = [], []
            # RoI pooling & build heap
            for ind, pair in enumerate(feature_pairs):
                pair = self.RoI(pair).view(-1)
                RoI_pairs.append(pair)
                attscore = self.attention(pair)
                # since this is a minheap, therefore we have a sign
                heappush(heap, [-attscore, ind])
            top_k_ind = {ind[1]:ind[0] for i in heapq.nsmallest(k, heap)}
            for ind, pair in enumerate(RoI_pairs):
                if ind in top_k_ind:
                    RoI_pairs[ind] *= top_k_ind[ind]
                else
                    RoI_pairs[ind] *= 0
            res_global.append(RoI_pairs)
            # output layer
        upper_res = self.fc6(res_global)
        return upper_res
    def test(person, bbox): # not sure the format of bbox
        cnt = 0
        print(len(person['joints']))
        for i in range(17):
            x = person['joints'][i*3]
            y = person['joints'][i*3+1]
            if point in bbox: cnt += 1
        if cnt >= 16
            return True
    def cut(person):
        targets, res = [1, 3, 4, 6, 7, 9, 10, 12, 13], []
        joints = person['joints']
        size = abs(joints[2*3]-joints[5*3])/2 # LShoulder to RShoulder
        for i in target:
            x = joints[i*3]
            y = joints[i*3+1]
            res.append([x+size, x-size, y+size, y-size])
        # Pelvis -> LHip to RHip
        x = (joints[8*3]+joints[11*3])/2
        y = (joints[8*3+1]+joints[11*3+1])/2
        res.append([x+size, x-size, y+size, y-size])
        return res
    def get_body_part(image_name='000000035005.jpg'):
        with open('results.json') as json_file:
            pics = json.load(json_file)
            pic = pics[image_name]
            people = pic['bodies']
            for i, person in enumerate(people):
                print(len(person))
                test(person, [0, 0, 0, 0])
                if test(person, bbox):
                    return cut(person)
    def get_body_pair():
        # reshape
