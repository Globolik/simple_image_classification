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
from CNN_veg_cnn import CNN, test_loader, train_loader
print('1')



# metric to check accuracy
def check_accuracy(loader, path):
    device = torch.device('cuda')
    model = CNN()
    model.cuda()
    PATH = path
    print('2')
    model.load_state_dict(torch.load(PATH))
    model.eval()

    if loader.dataset.train:
        print('checking training data')
    else:
        print('checking test data')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for data, targets, _ in loader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        print(f' got {num_samples}, correct {num_correct}, with accuracy {num_correct/num_samples:.2f}')

# check_accuracy(train_loader, path='model_16e.pth')
# 1e  got 14993, correct 9985, with accuracy 0.66
# 8e  got 14993, correct 12816, with accuracy 0.85
# 16e got 14993, correct 12483, with accuracy 0.83

check_accuracy(test_loader, path='model_12e_lr_decay_s2_g06.pth')
# 1e  got 2998, correct 2076, with accuracy 0.69
# 8e  got 2998, correct 2662, with accuracy 0.88
# 16e got 2998, correct 2518, with accuracy 0.84

def check_f1_for_1_class(loader, path):
    device = torch.device('cuda')
    model = CNN()
    model.cuda()
    PATH = path
    print('2')
    model.load_state_dict(torch.load(PATH))
    model.eval()
    with torch.no_grad():
        pred = np.ndarray(0)
        targ = np.ndarray(0)
        for data, targets, _ in loader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            # predictions is the indexes of max value in tensor (10 length so 0 to 9)
            _, prediction = scores.max(1)
            pred = np.append(pred, prediction.cpu().numpy())
            targ = np.append(targ, targets.cpu().numpy())
    print(classification_report(targ, pred))
check_f1_for_1_class(test_loader, path='model_12e_lr_decay_s2_g06.pth')
#8e               precision    recall  f1-score   support
#
#          0.0       0.86      0.87      0.87       199
#          1.0       0.82      0.87      0.84       200
#          2.0       0.96      0.99      0.98       200
#          3.0       0.95      0.93      0.94       200
#          4.0       0.90      0.79      0.84       200
#          5.0       0.88      0.91      0.89       200
#          6.0       0.91      0.83      0.87       200
#          7.0       0.86      0.85      0.85       200
#          8.0       0.92      0.81      0.86       200
#          9.0       0.87      0.94      0.90       200
#         10.0       0.90      0.99      0.94       200
#         11.0       0.89      0.87      0.88       200
#         12.0       0.90      0.90      0.90       200
#         13.0       0.83      0.90      0.87       199
#         14.0       0.87      0.87      0.87       200
#
#     accuracy                           0.89      2998
#    macro avg       0.89      0.89      0.89      2998
# weighted avg       0.89      0.89      0.89      2998
#
# 16e
#               precision    recall  f1-score   support
#
#          0.0       0.80   -  0.77   -  0.79       199
#          1.0       0.79      0.91      0.85       200
#          2.0       0.99      0.97      0.98       200
#          3.0       0.92      0.92      0.92       200
#          4.0       0.90   -  0.56   -  0.69       200
#          5.0       0.96   -  0.68      0.80       200
#          6.0       0.71      0.84      0.77       200
#          7.0       0.96   -  0.69      0.81       200
#          8.0       0.84      0.86      0.85       200
#          9.0       0.75      0.92      0.83       200
#         10.0       0.99      0.94      0.97       200
#         11.0       0.99      0.89      0.94       200
#         12.0       0.87      0.93      0.90       200
#         13.0       0.87   -  0.74      0.80       199
#         14.0       0.59      0.95   -  0.73       200
#
#     accuracy                           0.84      2998
#    macro avg       0.86      0.84      0.84      2998
# weighted avg       0.86      0.84      0.84      2998