# ################################################################
#                             HyperParameters
# ################################################################
# semi-supervised learning:
#     1. model structure
#     2. hype setting are important!

import torch
from torchvision.transforms import v2

class Hyperparameters:
    # ################################################################
    #                             Data
    # ################################################################
    mission_name = "galaxy classify"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #训练设备
    classes = ('edge_on_disk', 'eliptical', 'face_on_disk', 'merging')    #图像标签
    seed = 1234     #随机种子
    img_size = (128, 128)
    data_path = "../dataset_split"
    train_transform = v2.Compose([
        v2.CenterCrop((265, 265)),
        v2.Resize((160,160)),
        v2.RandomCrop(img_size),
        v2.RandomHorizontalFlip(),  # 随机翻转
        v2.RandomRotation(degrees=(0, 180)),#随机旋转
        v2.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.3),  # 随机调整brightness, contrast, saturation, hue
        # v2.TrivialAugmentWide(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #归一化
    ])
    val_transform = v2.Compose([
        v2.CenterCrop((212, 212)),
        v2.Resize(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # ################################################################
    #                             Exp
    # ################################################################
    batch_size = 128
    init_lr = 0.01 #learning rate 学习率
    epochs = 50     #训练轮数
    print_step = 100    #每次输出信息相隔的步数
    model_path = "../modelsave" #模型保存位置
    model_checkpoint = "../modelsave/model_38_0.534.pth" #模型继续训练位置；从头训练设置为None （不加双引号）

HP = Hyperparameters()