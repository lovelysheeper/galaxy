from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../1")
for i in range(10):
    writer.add_scalar("1", 0.1 * i, i)

writer.close()