# ---------------------------------------------
# 2019 DLP final project
# implementation of Pairwise Body-Part Attention for Recognizing Human-Object Interactions
# Written by Chihchia Li, alter027
# ---------------------------------------------

import os
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

import model.config as c

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

        if imageset == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((c.Full_img_x, c.Full_img_y)),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((c.Full_img_x,c.Full_img_y)),
                transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        image_name = self.img_files[index]
        image_path = os.path.join(self.path, image_name[0])
        assert os.path.exists(image_path)

        img = Image.open(image_path)
        orig_size = img.size
        # img.save('./test2.jpg')
        label = self.anno[:,index]

        return self.transform(img), label, image_name[0], orig_size

    def __len__(self):
        return len(self.img_files)