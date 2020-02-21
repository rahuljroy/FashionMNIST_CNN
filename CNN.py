import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms

opt_name = 'Adam'
lr = 0.002
l2_penalty = 0
momentum = None
nlayers = 3
epochs = 700


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def addnoise(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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


class CNN(nn.Module):
    
    def __init__(self, n_classes=10):
        dim_lastlayer = 20
        
        super(CNN, self).__init__()
        self.convlayer1 = nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1)
        self.convlayer2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1)
        self.linear1 = nn.Linear(64*7*7, dim_lastlayer)
        self.output = nn.Linear(dim_lastlayer, n_classes)
        self.optimizer = self.get_optimizer(opt_name, lr, l2_penalty, momentum=None)

    def num_features(self, x):
        size = x.size()[1:]
        num_features = 1
        
        for s in size:
            num_features *= s
        return num_features
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.convlayer1(x)), 2)
        x = F.max_pool2d(F.relu(self.convlayer2(x)), 2)
        x = x.view(-1, self.num_features(x))
        x = self.linear1(x)
        out = self.output(x)
        
        return out
    
    def get_optimizer(self, opt_name, lr, l2_penalty, momentum=None):
        if opt_name == 'SGD':
            return optim.SGD(
                self.parameters(), lr, weight_decay=l2_penalty)
        elif opt_name == 'Momentum':
            return optim.SGD(
                self.parameters(), lr=lr, momentum=momentum,
                weight_decay=l2_penalty)
        elif opt_name == 'Nesterov':
            return optim.SGD(
                self.parameters(), lr=lr, momentum=momentum,
                weight_decay=l2_penalty, nesterov=True)
        elif opt_name == 'Adagrad':
            return optim.Adagrad(
                self.parameters(), lr=lr, weight_decay=l2_penalty)
        elif opt_name == 'RMSProp':
            return optim.RMSprop(
                self.parameters(), lr=lr, weight_decay=l2_penalty)
        elif opt_name == 'Adam':
            return optim.Adam(
                self.parameters(), lr=lr, weight_decay=l2_penalty)


# Training the model

def train1epoch(model, trainloader, device):

    model.train()
    avg_loss = Average("average-loss")
    avg_accuracy = Average("average-accuracy")
    # addnoise = AddGaussianNoise()
    for batchIDx, (img, target) in enumerate(trainloader):
        img = Variable(img).to(device)
        # img = addnoise.addnoise(img)
        target = Variable(target).to(device)
        # print(img.shape)
        # zero out the gradients for a batch
        model.optimizer.zero_grad()
        
        #Forward propgation
        
        out = model(img)
        loss = F.cross_entropy(out, target)
        correctness = (target.data == out.max(dim=1)[1])
        accuracy = correctness.type(torch.FloatTensor).mean()
        
        # Backward propogation
        loss.backward()
        avg_loss.update(loss, img.shape[0])
        avg_accuracy.update(accuracy, img.shape[0])
        
        # Update the weights
        model.optimizer.step()

        return avg_loss.avg, avg_accuracy


if __name__ == "__main__":
    
    device = torch.device("cpu")
    
    model = CNN(10).to(device)

    
    model.optimizer = model.get_optimizer(opt_name, lr, l2_penalty, momentum)

    trans_img = transforms.Compose([transforms.ToTensor()])
    
    traindataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
    traindataset, validationdataset = random_split(traindataset, [50000,10000])
    trainloader = DataLoader(traindataset, batch_size=1024, shuffle=True)
    validationloader = DataLoader(validationdataset, batch_size=1024, shuffle=True)
    
    tracktrainloss = []
    tracktrainaccuracy = []
    trackvalidationloss = []
    trackvalidationaccuracy = []
    for i in range(epochs):
        trainloss, trainaccuracy = train1epoch(model, trainloader, device)
        tracktrainloss.append(trainloss)
        tracktrainaccuracy.append(trainaccuracy)
        validationloss, validationaccuracy = train1epoch(model, validationloader, device)
        trackvalidationloss.append(validationloss)
        trackvalidationaccuracy.append(validationaccuracy)
        if (i%10 ==0):
            print("epoch: ", i, " Loss: ", trainloss, " Accuracy: ", trainaccuracy)

    plt.figure()
    plt.plot(tracktrainloss)
    plt.plot(trackvalidationloss)
    plt.title("Training Loss MLNN")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    filename = "Models/CNN/" + '{0}_{1}_{2}_{3}.pt'.format(opt_name, lr, nlayers, l2_penalty)
    torch.save(model.state_dict(), filename)