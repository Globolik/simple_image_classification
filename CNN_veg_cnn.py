import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from CNN_veg_dataset import VegitablesDataset




class CNN(nn.Module):
    def __init__(self, in_cannels=3, num_classes=15):
        super(CNN, self).__init__()
        # images is 224x224x3 for each out_channels conv1 will have a filter. each filter will have in_cannels
        # kernels. Output will have out_channels channels. if groups = in_channels each channel get
        # out_channels/in_channels filters and then output is concat
        self.conv1 = nn.Conv2d(in_channels=in_cannels, out_channels=6, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3))
        # output
        # torch.Size([32, 6, 112, 112])
        # params
        # conv1.weight torch.Size([6, 3, 7, 7])
        # conv1.bias torch.Size([6])

        self.bn_cv1 = nn.BatchNorm2d(num_features=6)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # 56x56
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
        self.bn_cv2 = nn.BatchNorm2d(num_features=12)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(12*14*14, out_features=num_classes)

    def forward(self, x):
        x = self.bn_cv1(self.conv1(x))
        x = F.relu(x)
        x = self.pool(x)  # torch.Size([32, 6, 56, 56])
        x = self.bn_cv2(self.conv2(x))
        x = F.relu(x)
        x = self.pool(x)  # torch.Size([32, 12, 14, 14])
        x = x.reshape(x.shape[0], -1)  # torch.Size([32, 2352])
        x = self.fc1(x)
        return x


# cnn = CNN()
#
# for name, param in cnn.named_parameters():
#     print(name, param.size())
#
# model = CNN()
# x = torch.rand(32, 3, 224, 224)
# print(model.forward(x).shape)

# set device
device = torch.device('cuda')
# Hyperparameters
lr = 0.001
batch_size = 64
num_epochs = 12





mean = (119, 117, 87)
std = (51, 51, 49)
mean = tuple(round(_ / 255, 5) for _ in mean)
std = tuple(round(_ / 255, 5) for _ in std)

transforms_to_do = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomApply(
            [
                transforms.RandomCrop((190, 190)),
                transforms.RandomRotation(degrees=15,),  # 30
                transforms.ColorJitter(brightness=0.1),
                transforms.Resize((224, 224)),

            ],
            p=0.5
        ),

        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

transforms_test = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

dataset = VegitablesDataset(
    csv_annotations='annotations.csv',
    root_dir='/home/globolik/simple_torch_nn/vegetables/Vegetable Images',
    transform=transforms_to_do,
    train=True

)




# load_data
train_dataset = VegitablesDataset(
    csv_annotations='annotations.csv',
    root_dir='/home/globolik/simple_torch_nn/vegetables/Vegetable Images',
    transform=transforms_to_do,
    train=True

)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = VegitablesDataset(
    csv_annotations='annotations_test.csv',
    root_dir='/home/globolik/simple_torch_nn/vegetables/Vegetable Images',
    transform=transforms_test,
    train=False

)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    from torchsummary import summary
    # init model
    model = CNN()
    model.cuda()
    summary(model, input_size=(3, 224, 224))
    #         Layer (type)               Output Shape         Param #
    # ================================================================
    #             Conv2d-1          [-1, 6, 112, 112]             888
    #        BatchNorm2d-2          [-1, 6, 112, 112]              12
    #          MaxPool2d-3            [-1, 6, 56, 56]               0
    #             Conv2d-4           [-1, 12, 29, 29]           1,164
    #        BatchNorm2d-5           [-1, 12, 29, 29]              24
    #          MaxPool2d-6           [-1, 12, 14, 14]               0
    #             Linear-7                   [-1, 15]          35,295
    # ================================================================
    # Total params: 37,383
    # Trainable params: 37,383
    # Non-trainable params: 0
    # ----------------------------------------------------------------
    # Input size (MB): 0.57
    # Forward/backward pass size (MB): 1.46
    # Params size (MB): 0.14
    # Estimated Total Size (MB): 2.18
    # ----------------------------------------------------------------

    # loss and optimizer
    criter = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8, last_epoch=8)
    # train
    for epoch in range(num_epochs):
        print(f'epoch {epoch}')
        for batch_idx, (data, targets, _) in enumerate(train_loader):
            print(f'batch_idx {batch_idx}')
            # data to cuda
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward
            scores = model(data)
            loss = criter(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # opt step
            optimizer.step()
        scheduler.step()
    PATH = 'model_12e_lr_decay_s2_g06.pth'
    torch.save(model.state_dict(), PATH)
