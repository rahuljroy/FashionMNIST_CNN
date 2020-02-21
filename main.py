import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms


# Parameters of MLNN
opt_name_MLNN = 'RMSProp'
lr_MLNN = 0.001
l2_penalty_MLNN = 0
momentum_MLNN = None
nlayers_MLNN = 3

# Parameters of CNN
opt_name_CNN = 'Adam'
lr_CNN = 0.002
l2_penalty_CNN = 0
momentum_CNN = None
nlayers_CNN = 3


class Average(object):
    
    def __init__(self, name, fmt = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def test(model, testloader):
    """ Training the model using the given dataloader for 1 epoch.
    Input: Model, Dataset, optimizer,
    """

    model.eval()
    avg_loss = Average("average-loss")

    y_gt = []
    y_pred_label = []

    for batch_idx, (img, y_true) in enumerate(testloader):
        img = Variable(img)
        y_true = Variable(y_true)
        out = model(img)
        y_pred = F.softmax(out, dim=1)
        y_pred_label_tmp = torch.argmax(y_pred, dim=1)

        loss = F.cross_entropy(out, y_true)
        avg_loss.update(loss, img.shape[0])

        # Add the labels
        y_gt += list(y_true.numpy())
        y_pred_label += list(y_pred_label_tmp.numpy())

    return avg_loss.avg, y_gt, y_pred_label

if __name__ == "__main__":

    device = torch.device('cpu')
    
    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
    testloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    from MLNN import MLNN
    model_MLP = MLNN(10).to(device)
    file1 = "Models/MLNN/" + '{0}_{1}_{2}_{3}.pt'.format(opt_name_MLNN, lr_MLNN, nlayers_MLNN, l2_penalty_MLNN)
    model_MLP.load_state_dict(torch.load(file1))

    from CNN import CNN
    model_conv_net = CNN(10).to(device)
    file2 = "Models/CNN/" + '{0}_{1}_{2}_{3}.pt'.format(opt_name_CNN, lr_CNN, nlayers_CNN, l2_penalty_CNN)
    model_conv_net.load_state_dict(torch.load(file2))

    loss, gt, pred = test(model_MLP, testloader)
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

    loss, gt, pred = test(model_conv_net, testloader)
    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))