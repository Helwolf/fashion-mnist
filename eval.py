import numpy as np
import torch
from data_util import get_train_data, get_test_data, create_result
from ResNet import resnet18
from torch.utils.data import DataLoader
from dataset import FashionMnist
import argparse
from sklearn.metrics import classification_report
from torch_tools import worker_init


worker_init()

def get_result(model, dataset):

    result = []
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    model.eval()
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            res = model(inputs)
            _, pred = torch.max(res, 1)
            result.extend(list(pred.cpu().numpy()))
    return np.array(result)

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--gpu-id', type=int, default=0, metavar='G',
                    help='GPU ID (default: 0)')

parser.add_argument('--model', type=str, default='0.9396_model.ckpt', metavar='M')
args = parser.parse_args()

model = resnet18()
wt = torch.load('result/{}'.format(args.model))
model.load_state_dict(wt)
device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
model = model.to(device)
_, val_x, _, val_y = get_train_data(test_size=0.1, one_hot=False)
test_data = get_test_data()

val_dataset = FashionMnist(data_type='test', data=val_x)
test_dataset = FashionMnist(data_type='test', data=test_data)

val_result = get_result(model, val_dataset)
test_result = get_result(model, test_dataset)
print('Accuracy is {}'.format(np.mean(val_result==val_y)))
report = classification_report(val_y, val_result)

heat_map = np.zeros((10, 10))
for i in range(5000):
    t = val_y[i]
    p = val_result[i]
    heat_map[t][p] += 1
np.save('heatmap', heat_map)

print(report)
create_result(test_result)
