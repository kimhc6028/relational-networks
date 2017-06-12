import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class RN(nn.Module):

    def __init__(self,args):
        super(RN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)
        
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)
        #self.g_fc1 = nn.Linear(24*5*5+11, 256)
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)
        self.f_fc2 = nn.Linear(256, 256)
        self.f_fc3 = nn.Linear(256, 10)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        self.coord_lst = [torch.from_numpy(np.array([self.cvt_coord(i) for _ in range(args.batch_size)])) for i in range(25)]


    def cvt_coord(self, i):
        return [(i/5-2)/2., (i%5-2)/2.]


    def forward(self, img, qst):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        ## x = (64 x 256 x 5 x 5)
        """g"""
        x_g = 0
        for i in range(25):
            oi = x[:,:,i/5,i%5]
            self.coord_oi.data.copy_(self.coord_lst[i])
            oi = torch.cat((oi,self.coord_oi), 1)
            for j in range(25):
                oj = x[:,:,j/5,j%5]
                self.coord_oj.data.copy_(self.coord_lst[j])
                oj = torch.cat((oj,self.coord_oj), 1)
                x_ = torch.cat((oi,oj,qst), 1)
                x_ = self.g_fc1(x_)
                x_ = F.relu(x_)
                x_ = self.g_fc2(x_)
                x_ = F.relu(x_)
                x_ = self.g_fc3(x_)
                x_ = F.relu(x_)
                x_ = self.g_fc4(x_)
                x_ = F.relu(x_)
                x_g += x_
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = F.dropout(x_f)
        x_f = self.f_fc3(x_f)

        return F.log_softmax(x_f)


    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy
        

    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy


    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}.pth'.format(epoch))
