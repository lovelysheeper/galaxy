import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time
from argparse import ArgumentParser

from model import *
from config import HP
from utils import *

# 训练的设备
print("device:{}".format(HP.device))


# seed init
torch.manual_seed(HP.seed)
torch.cuda.manual_seed(HP.seed)

# 准备数据集
train_data = torchvision.datasets.ImageFolder(root=os.path.join(HP.data_path, "train"), transform=HP.train_transform) #加载训练集
val_data = torchvision.datasets.ImageFolder(root=os.path.join(HP.data_path, "val"), transform=HP.val_transform) #加载验证集

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=HP.batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=HP.batch_size, shuffle=False)
# 添加TensorBoard
writer = SummaryWriter(log_dir="../logs")

# length 长度
train_data_size = len(train_data)
val_data_size = len(val_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(val_data_size))

# shadow ema model
def new_ema_model():
    model = Vgg16_net()
    model = model.to(HP.device)
    for param in model.parameters():
        param.detach_() # disable gradient trace
    return model


# save func
def save_checkpoint(model_, ema_model_, epoch_, optm, checkpoint_path):
    save_dict = {
        'epoch': epoch_,
        'model_state_dict': model_.state_dict(),
        'ema_model_state_dict': ema_model_.state_dict(),
        'optimizer_state_dict': optm.state_dict(),
    }
    torch.save(save_dict, checkpoint_path)

# train func
def train():
    parser = ArgumentParser(description='Model Training')
    parser.add_argument(
        '--c',
        default=HP.model_checkpoint,
        type=str,
        help='train from scratch or resume from checkpoint'
    )
    args = parser.parse_args()

    # new models: model/ema_model and WeightEMA instance
    model = Vgg16_net()
    model = model.to(HP.device)
    ema_model = new_ema_model()
    model_ema_opt = WeightEMA(model, ema_model)

    # 损失函数
    criterion = nn.CrossEntropyLoss().to(HP.device)

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=HP.init_lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=HP.init_lr, weight_decay=0.001)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    iters = len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)

    # 设置训练网络的一些参数
    # 记录训练的次数
    start_epoch = 0
    total_train_step = 0
    total_val_step = 0

    if args.c:
        checkpoint = torch.load(args.c)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('Resume From %s.' % args.c)
    else:
        print('Training from scratch!')

    best_acc = 0.0
    # train loop
    start_time = time.time()
    for epoch in range(start_epoch, HP.epochs):
        print('Start epoch: {}'.format(epoch))
        print("--------第{}轮训练开始--------".format(epoch + 1))

        model.train()   #训练
        total_train_accuracy = 0
        for batch, data in enumerate(train_dataloader):
            imgs, targets = data
            imgs = imgs.to(HP.device)
            targets = targets.to(HP.device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            accuracy = (outputs.argmax(1) == targets).sum()
            total_train_accuracy = total_train_accuracy + accuracy

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model_ema_opt.step()
            # scheduler.step(i + batch / iters)


            total_train_step = total_train_step + 1
            if total_train_step % HP.print_step == 0:
                end_time = time.time()
                print("time:{}".format(end_time - start_time))
                print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        total_train_accuracy = total_train_accuracy / train_data_size
        print("训练集的正确率：{}".format(total_train_accuracy))
        writer.add_scalar("train_accuracy", total_train_accuracy, total_train_step)

        model.eval()    #验证
        total_val_loss = 0
        total_val_accuracy = 0
        with torch.no_grad():
            for data in val_dataloader:
                imgs, targets = data
                imgs = imgs.to(HP.device)
                targets = targets.to(HP.device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                total_val_loss = total_val_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_val_accuracy = total_val_accuracy + accuracy
        total_val_accuracy = total_val_accuracy / val_data_size
        print("整体验证集上的Loss：{}".format(total_val_loss))
        print("整体验证集上的正确率：{}".format(total_val_accuracy))
        writer.add_scalar("val_loss", total_val_loss, total_val_step)
        writer.add_scalar("val_accuracy", total_val_accuracy, total_val_step)
        total_val_step = total_val_step + 1

        scheduler.step()

        if total_val_accuracy > best_acc:
            best_acc = total_val_accuracy
            model_name = 'model_{}_{}.pth'.format(epoch, '%.3f' % best_acc)
            save_checkpoint(model, ema_model, epoch, optimizer, os.path.join(HP.model_path, model_name))
    writer.close()

if __name__ == '__main__':
    train()

