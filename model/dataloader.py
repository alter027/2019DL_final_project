# ---------------------------------------------
# 2019 DLP final project
# implementation of Pairwise Body-Part Attention for Recognizing Human-Object Interactions
# Written by Chihchia Li, alter027
# ---------------------------------------------

import os
import scipy
import scipy.io as sio
import torch
from torch.utils.data.dataset import Dataset

class HICO(Dataset):
    def __init__(self, imageset='train', root='/home/pj19/twoTFolder/DL_final_project/data/hico_20150920/'):
        self.imageset = imageset
        self.path = os.path.join(root,'images','{}2015'.format(imageset))

        # result for deformable convnet? img feature
        anno_file = os.path.join(root+'anno.mat')
        ld = sio.loadmat(anno_file)
        files = ld['list_'+imageset]
        anno = ld['anno_'+imageset]

        self.img_files = [i[0] for i in files]
        self.anno = anno

    def __getitem__(self, index):
        image_name = self.img_files[index]
        image_path = os.path.join(self.path, image_name[0])
        assert os.path.exists(image_path)

        img = scipy.misc.imread(image_path, mode='RGB')
        label = self.anno[:,index]

        return img, label

    def __len__(self):
        return len(self.img_files)