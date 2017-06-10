"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import os
import cPickle as pickle
import random
import numpy as np

import torch
#import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd import Variable

import model
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = model.RN(args)
model_dirs = './model'
bs = args.batch_size
input_img = torch.FloatTensor(bs, 3, 75, 75)
input_qst = torch.FloatTensor(bs, 10)
label = torch.LongTensor(bs)

if args.cuda:
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)


def cvt_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)

    
def train(epoch, rel, norel):
    model.train()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    for batch_idx in range(len(rel[0]) / bs):
        cvt_data(rel, batch_idx)
        accuracy_rel = model.train_(input_img, input_qst, label)

        cvt_data(norel, batch_idx)
        accuracy_norel = model.train_(input_img, input_qst, label)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(epoch, batch_idx * bs * 2, len(rel[0]) * 2, \
                                                                                                                           100. * batch_idx * bs/ len(rel[0]), accuracy_rel, accuracy_norel))
            

def test(epoch, rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    accuracy_rels = []
    accuracy_norels = []
    for batch_idx in range(len(rel[0]) / bs):
        cvt_data(rel, batch_idx)
        accuracy_rels.append(model.test_(input_img, input_qst, label))

        cvt_data(norel, batch_idx)
        accuracy_norels.append(model.test_(input_img, input_qst, label))

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Test set: Relation accuracy: {:.0f}% | Non-relation accuracy: {:.0f}%\n'.format(
        accuracy_rel, accuracy_norel))

    
def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    f = open(filename, 'r')
    train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    print('converting data...')
    datasets = [rel_train, rel_test, norel_train, norel_test]
    for dataset in datasets:
        random.shuffle(dataset)
    n_datasets = []
    for dataset in datasets:
        img = [e[0] for e in dataset]
        qst = [e[1] for e in dataset]
        ans = [e[2] for e in dataset]
        n_datasets.append((img,qst,ans))

    return tuple(n_datasets)

rel_train, rel_test, norel_train, norel_test = load_data()

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

for epoch in range(1, args.epochs + 1):
    train(epoch, rel_train, norel_train)
    test(epoch, rel_test, norel_test)
    model.save_model(epoch)
