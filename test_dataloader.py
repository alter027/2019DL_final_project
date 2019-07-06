import torch
import torchvision.transforms
from dataloader import HICO

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    normalize,
])
train_set = HICO('data/hico_20150920', 'train')
print(train_set)
