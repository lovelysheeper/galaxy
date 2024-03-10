import os
import torchvision
from torch.utils.data import DataLoader
from model import *
from config import *

model_test = Vgg16_net()
checkpoint = torch.load("../modelsave/model_43_0.908.pth")
model_test.load_state_dict(checkpoint['ema_model_state_dict'])
if torch.cuda.is_available():
    model_test = model_test.cuda()

test_data = torchvision.datasets.ImageFolder(root="../predict", transform=v2.Compose([
                                                v2.Resize(HP.img_size),
                                                v2.ToImage(),
                                                v2.ToDtype(torch.float32, scale=True),
                                                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])) #加载预测图像
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

model_test.eval()
with torch.no_grad():
    print("依次对应的类别为：\n")
    for data in test_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        outputs = model_test(imgs)
        print(HP.classes[outputs.argmax(1)], "\n")
