import os

import torchvision
from torch.utils.data import DataLoader

from utils import *
from model import *
from config import *

model_test = Vgg16_net()
checkpoint = torch.load("../modelsave/model_38_0.922.pth")
model_test.load_state_dict(checkpoint['ema_model_state_dict'])
if torch.cuda.is_available():
    model_test = model_test.cuda()

test_data = torchvision.datasets.ImageFolder(root=os.path.join(HP.data_path, "test"), transform=HP.val_transform) #加载验证集
test_dataloader = DataLoader(test_data, batch_size=HP.batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss().to(HP.device)


model_test.eval()
total_test_loss = 0
total_accuracy = 0
test_real_labels = []
test_pre_labels = []

with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = model_test(imgs)
        loss = criterion(outputs, targets)
        total_test_loss = total_test_loss + loss.item()
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy = total_accuracy + accuracy
        target_numpy = targets.cpu().numpy()
        y_pred = torch.softmax(outputs, dim=1)
        y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
        test_real_labels.extend(target_numpy)
        test_pre_labels.extend(y_pred)

print("正确率：{}".format(total_accuracy/len(test_data)))
class_names_length = len(HP.classes)
heat_maps = np.zeros((class_names_length, class_names_length))
for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
    heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1
heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
heat_maps_float = heat_maps / heat_maps_sum
show_heatmaps(title="{}_confusion_matrix".format(HP.mission_name), x_labels=HP.classes, y_labels=HP.classes, harvest=heat_maps_float,
              save_name="../output/heatmap_{}.png".format(HP.mission_name))
print("heatmap已生成")