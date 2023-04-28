'''Train with PyTorch.'''
from __future__ import print_function
import os
import argparse
import utils
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import transforms as transforms
from torch.autograd import Variable
from models import *
from datetime import datetime
from BC_data import MyDataset
# from BC_data_rgb import MyDataset
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='Resnet18', help='CNN architecture, Resnet18')
parser.add_argument('--dataset', type=str, default='BC', help='CNN architecture')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate') # 0.01, 0.005
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--save_dir', type=str, default='result_model', help='save log and model')
opt = parser.parse_args()

# SEED = 2
SEED = 4
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
print('seed:',torch.initial_seed())

Dataroot = 'data/BC650_jpg_scale'
# Dataroot_val = 'data/seg_val_jpg'
Dataroot_val = 'data/seg_val2_jpg'

use_cuda = torch.cuda.is_available()
best_Test_auc_epoch = 0; best_Test_acc = 0; best_Test_auc = 0  # best PublicTest accuracy
best_val_auc_epoch = 0; best_val_acc = 0; best_val_auc = 0  # best Private accuracy
prob_all_tr = []; label_all_tr = []
prob_all_test = []; label_all_test = []

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 20

path = os.path.join(opt.save_dir)
if not os.path.isdir(path): os.mkdir(path)
results_log_csv_name = opt.save_dir+'results_log.csv'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.RandomCrop(64),
    # transforms.Resize((64, 64)),
    # transforms.RandomCrop(64, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2520], std=[0.1694])]) #BC650 resize70
    # transforms.Normalize(mean=[0.2475], std=[0.1681])]) #BC650 resize68
    # transforms.Normalize(mean=[0.2381], std=[0.1653])]) #BC650

transform_test = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2520], std=[0.1694])]) #BC650 resize70

transform_val = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2152], std=[0.1575])]) #val2 resize70
    # transforms.Normalize(mean=[0.2118], std=[0.1556])]) #val2 resize68
    # transforms.Normalize(mean=[0.2443], std=[0.1834])]) #val3


#train and test data
train_data = MyDataset(split = 'train', data_dir=Dataroot, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=opt.bs, shuffle=True,
                                          num_workers=1, pin_memory=False, drop_last=True)
test_data = MyDataset(split = 'test', data_dir=Dataroot, transform=transform_test)
testloader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False,
                                         num_workers=1, pin_memory=False)
#external val data
exter_val_data = MyDataset(split = 'val', data_dir=Dataroot_val, transform=transform_val)
exter_valloader = torch.utils.data.DataLoader(exter_val_data, batch_size=8, shuffle=False,
                                         num_workers=1, pin_memory=False)
# Model
if opt.model == 'VGG19': net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    # net = ResNet18()
    net = torchvision.models.resnet18(pretrained=False)
    net.load_state_dict(torch.load('./result_model/resnet18.pth'))
    # net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 2)
    # net.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(net.fc.in_features, 2))


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch,eta_min=1e-6)
softmax_func = nn.Softmax(dim=1)

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'Test_model.t7'))
    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['acc']
    best_Test_auc_epoch = checkpoint['epoch']
    start_epoch = checkpoint['epoch'] + 1
    for x in range(start_epoch): scheduler.step()
else: print('==> Building model..')

if use_cuda: net = net.cuda()

def find_optimal_cutoff(tpr,fpr,threshold):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold

def std_c_matrix(fpr, tpr, thresholds, true_label, model_pred):
    cutoff = find_optimal_cutoff(tpr, fpr, thresholds)
    y_pred = list(map(lambda x: 1 if x >= cutoff else 0, model_pred))
    c_matrix = confusion_matrix(true_label, y_pred)
    c_matrix = c_matrix / c_matrix.sum(axis=1)[:, np.newaxis]
    # # return c_matrix
    return np.around(c_matrix, 3)

def print_roc(fpr,tpr,thresholds):
    print('fpr:')
    for value in fpr: print(value)
    print('tpr:')
    for value in tpr: print(value)
    print('thresholds:')
    for value in thresholds: print(value)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc, Train_auc, prob_all_tr, label_all_tr
    net.train()
    train_loss, correct, total = 0, 0, 0
    prob_all_tr = []
    label_all_tr = []
    scheduler.step()
    print('learning_rate: %s' % str(scheduler.get_lr()))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        #compute AUC
        prob_all_tr.extend(softmax_func(outputs)[:, 1].detach().numpy())
        label_all_tr.extend(targets)
        utils.progress_bar(batch_idx, len(trainloader), 'Train: | Loss: %.3f | Acc: %.3f (%d/%d)'
            % (train_loss/(batch_idx+1), float(correct)/total, correct, total))

    Train_acc = float(correct)/total #*100.
    Train_auc = roc_auc_score(label_all_tr, prob_all_tr)
    print('Train_auc:%0.3f'% Train_auc)

def val(epoch):
    global val_acc, best_val_acc, best_val_auc, best_val_auc_epoch, prob_all_tr, label_all_tr, \
    prob_all_test, label_all_test, prob_all_tr_ord, label_all_tr_ord, Train_acc_ord, Train_auc_ord, \
    prob_all_val, label_all_val
    net.eval()
    val_loss = 0; correct = 0; total = 0
    prob_all_val, label_all_val = [], []
    prob_all_tr_ord, label_all_tr_ord = [], []
    for batch_idx, (inputs, targets) in enumerate(exter_valloader):
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # compute AUC
        prob_all_val.extend(softmax_func(outputs)[:, 1].detach().numpy())
        label_all_val.extend(targets)
        utils.progress_bar(batch_idx, len(exter_valloader), 'Val: | Loss: %.3f | Acc: %.3f (%d/%d)'
                           % (val_loss / (batch_idx + 1), float(correct) / total, correct, total))
    # Save checkpoint.
    val_acc = float(correct) / total  # *100.
    val_auc = roc_auc_score(label_all_val, prob_all_val)
    print('val_auc: %0.3f' % val_auc)
    # if val_acc > best_val_acc: best_val_acc = val_acc
    #     print("best_val_acc: %0.3f" % val_acc)
    if val_auc > best_val_auc:
        # print('Saving model..')
        # print("best_val_auc: %0.3f" % val_auc)
        # state = {'net': net.state_dict() if use_cuda else net, 'acc': val_acc, 'epoch': epoch}
        # if not os.path.isdir(path): os.mkdir(path)
        # torch.save(state, os.path.join(path, 'Ex_Val_model.t7'))
        best_val_acc = val_acc
        best_val_auc = val_auc
    # Compute Sensitivity and Specificity
    fpr, tpr, thresholds = roc_curve(label_all_tr, prob_all_tr, pos_label=1)
    senspe = tpr + (1 - fpr)
    print('Train Sensitivity:{:.3f}, Specificity:{:.3f}'.format(tpr[np.argmax(senspe)], (1 - fpr)[np.argmax(senspe)]))
    # print_roc(fpr, tpr, thresholds)
    fpr, tpr, thresholds = roc_curve(label_all_test, prob_all_test, pos_label=1)
    senspe = tpr + (1 - fpr)
    print('Test Sensitivity:{:.3f}, Specificity:{:.3f}'.format(tpr[np.argmax(senspe)], (1 - fpr)[np.argmax(senspe)]))
    # print_roc(fpr, tpr, thresholds)
    fpr, tpr, thresholds = roc_curve(label_all_val, prob_all_val, pos_label=1)
    senspe = tpr + (1 - fpr)
    print('Val Sensitivity:{:.3f}, Specificity:{:.3f}'.format(tpr[np.argmax(senspe)], (1 - fpr)[np.argmax(senspe)]))
    # print_roc(fpr, tpr, thresholds)


def test(epoch):
    global Test_acc, best_Test_acc, best_Test_auc, best_Test_auc_epoch, prob_all_test, label_all_test
    net.eval()
    Test_loss = 0; correct = 0; total = 0
    prob_all_test = []
    label_all_test = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        Test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # compute AUC
        prob_all_test.extend(softmax_func(outputs)[:, 1].detach().numpy())
        label_all_test.extend(targets)
        utils.progress_bar(batch_idx, len(testloader), 'Test: | Loss: %.3f | Acc: %.3f (%d/%d)'
                           % (Test_loss / (batch_idx + 1), float(correct) / total, correct, total))
    Test_acc = float(correct)/total #*100.
    # Save checkpoint.
    # if Test_acc > best_Test_acc: best_Test_acc = Test_acc
    #     print("best_Test_acc: %0.3f" % Test_acc)
    Test_auc = roc_auc_score(label_all_test, prob_all_test)
    print("Test_auc: %0.3f" % Test_auc)
    if Test_auc > best_Test_auc:
        print('Saving model..')
        print("best_Test_auc: %0.3f" % Test_auc)
        state = {'net': net.state_dict() if use_cuda else net, 'acc': Test_acc, 'epoch': epoch}
        if not os.path.isdir(path): os.mkdir(path)
        torch.save(state, os.path.join(path, 'Inter_Test_model.t7'))
        best_Test_auc = Test_auc
        best_Test_auc_epoch = epoch
        best_Test_acc = Test_acc
        # Val
        val(epoch)


if __name__ == '__main__':
    #record train log
    with open('data/CNN_ROC_results_log.csv', 'w') as f_roc: f_roc.write('record fpr, tpr, thresholds and  \n')
    with open('data/CNN_score_results_log_tr.csv', 'w') as f_tr: f_tr.write('train score \n')
    with open('data/CNN_score_results_log_te.csv', 'w') as f_te: f_te.write('test score \n')
    with open('data/CNN_score_results_log_val.csv', 'w') as f_val: f_val.write('val score \n')
    #start train
    for epoch in range(start_epoch, total_epoch):
        print('time:',datetime.now().strftime('%b%d-%H:%M:%S'),'(',opt.save_dir,')')
        train(epoch)
        test(epoch)
        # val(epoch)
        # Log results
        # with open(os.path.join(path, results_log_csv_name), 'a') as f:
        #     f.write('%03d,%0.3f,%0.3f,%s,\n' % (epoch, Train_acc, Test_acc,datetime.now().strftime('%b%d-%H:%M:%S')))

    # record data
    with open('data/CNN_score_results_log_tr.csv', 'a') as f_tr:
        for prob in prob_all_tr_ord: f_tr.write('%0.6f\n' % (prob))
    with open('data/CNN_score_results_log_te.csv', 'a') as f_te:
        for prob in prob_all_test: f_te.write('%0.6f\n' % (prob))
    with open('data/CNN_score_results_log_val.csv', 'a') as f_val:
        for prob in prob_all_val: f_val.write('%0.6f\n' % (prob))
    # ROC
    fpr, tpr, thresholds = roc_curve(label_all_tr_ord, prob_all_tr_ord, pos_label=1)
    c_matrix = std_c_matrix(fpr, tpr, thresholds, label_all_tr_ord, prob_all_tr_ord)
    with open('data/CNN_ROC_results_log.csv', 'a') as f_roc:
        f_roc.write('record Train Set fpr, tpr, thresholds \n')
        f_roc.write('fpr:,%s\n' % (str(fpr)))
        f_roc.write('tpr:,%s\n' % (str(tpr)))
        f_roc.write('thresholds:,%s\n' % (str(thresholds)))
        f_roc.write('train c_matrix:,%s\n' % (str(c_matrix)))
    fpr, tpr, thresholds = roc_curve(label_all_test, prob_all_test, pos_label=1)
    c_matrix = std_c_matrix(fpr, tpr, thresholds, label_all_test, prob_all_test)
    with open('data/CNN_ROC_results_log.csv', 'a') as f_roc:
        f_roc.write('record Test Set fpr, tpr, thresholds \n')
        f_roc.write('fpr:,%s\n' % (str(fpr)))
        f_roc.write('tpr:,%s\n' % (str(tpr)))
        f_roc.write('thresholds:,%s\n' % (str(thresholds)))
        f_roc.write('test c_matrix:,%s\n' % (str(c_matrix)))
    fpr, tpr, thresholds = roc_curve(label_all_val, prob_all_val, pos_label=1)
    c_matrix = std_c_matrix(fpr, tpr, thresholds, label_all_val, prob_all_val)
    with open('data/CNN_ROC_results_log.csv', 'a') as f_roc:
        f_roc.write('record Val Set fpr, tpr, thresholds \n')
        f_roc.write('fpr:,%s\n' % (str(fpr)))
        f_roc.write('tpr:,%s\n' % (str(tpr)))
        f_roc.write('thresholds:,%s\n' % (str(thresholds)))
        f_roc.write('val c_matrix:,%s\n' % (str(c_matrix)))
    print("best_Test_acc: %0.3f, best_Test_auc: %0.3f, best_Test_auc_epoch: %d" % (best_Test_acc,
                                                            best_Test_auc, best_Test_auc_epoch))
    print("best_val_acc: %0.3f, best_val_auc: %0.3f, best_val_auc_epoch: %d" % (best_val_acc,
                                                            best_val_auc, best_val_auc_epoch))


