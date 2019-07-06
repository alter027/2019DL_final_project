import model.vgg as vgg
import cv2
from torchvision import transforms

model = vgg.vgg16().cuda(2)
transform = transforms.Compose([
    transforms.ToTensor(),
])
data = cv2.imread('input_img/000000035005.jpg')
data = transform(data).cuda(2)
print(data.shape)
print(model(data.unsqueeze_(0)))