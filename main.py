import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_util import get_train_data
from torch.utils.data import DataLoader
from train_util import train_model
from torchsummary import summary
from torch_tools import worker_init
import argparse
from ResNet import resnet18
from WRN import wrn
from dataset import FashionMnist
from basic_model import BasicModle
from torchvision.datasets import MNIST
from torchvision import transforms

parser = argparse.ArgumentParser(description='PyTorch')

parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                    help='learning rate (default: 0.003)')
parser.add_argument('--gpu-id', type=int, default=0, metavar='G',
                    help='GPU ID (default: 0)')
args = parser.parse_args()

worker_init()


device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

# train_x, val_x, train_y, val_y = get_train_data(test_size=0.1, one_hot=False)
train_data = MNIST('.', download=True, transform=transforms.ToTensor())
val_data = MNIST('.', train=False, transform=transforms.ToTensor())
# train_data = FashionMnist(data_type='train', data=(train_x, train_y))
# val_data = FashionMnist(data_type='val', data=(val_x, val_y))
datasets = {'train': train_data, 'val': val_data}
dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, num_workers=4)
               for x in ['train', 'val']}


# model = wrn(depth=28, num_classes=10, widen_factor=8, drop_rate=0.5)
model = resnet18()
# model = BasicModle()
# summary(model.to(device), (1, 28, 28))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.Adadelta(model.parameters(), weight_decay=0.02)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
exp_lr_scheduler = None
model, best_acc = train_model(model=model,
                              device=device,
                              dataloaders=dataloaders,
                              criterion=criterion, optimizer=optimizer,
                              scheduler=exp_lr_scheduler,
                              epochs=args.epochs)

torch.save(model.state_dict(), 'result/{}_model.ckpt'.format(best_acc))

