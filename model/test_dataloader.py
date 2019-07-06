import torch
import torchvision.transforms
from dataloader import HICO

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    normalize,
])
train_set = HICO('train', 'data/hico_20150920/')
print(train_set)
print(len(train_set))

for i,j in train_set:
    print(i,j)
    input()