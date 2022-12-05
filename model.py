import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy

# RN_INPUT_SIZE = 49    # CNN
# NUM_OBJECTS = 24        # CNN

RN_INPUT_SIZE = 10    # State desc task
NUM_OBJECTS = 6         # State desc task


class ConvInputModel(nn.Module):
    # This class creates the CNN
    def __init__(self):
        super(ConvInputModel, self).__init__()

        # Define the four convolutional layers
        # Define batch normalization after each conv layer
        # We modified two of the convolutional layers to improve performance
        self.conv1 = nn.Conv2d(3, 24, 3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 4, stride=3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)
        # self.conv5 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        # self.batchNorm5 = nn.BatchNorm2d(24)
        
    def forward(self, img):
        # Forward pass of the convolution
        # After each convolution, perform a ReLu transformation and batch normalization
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
        # x = self.conv5(x)
        # x = F.relu(x)
        # x = self.batchNorm5(x)
        return x

  
class FCOutputModel(nn.Module):
    # This is the final pass to convert the output to a vector over the possible answers
    # Produce logits and perform softmax (F.log_softmax)
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # In the forward pass, in addition to the two fully connected layers,
        # there is also a dropout between the layers.
        # Dropouts randomly zero some values as a form of regluarization and prevent overfitting.
        # Overfitting was not a huge concern for us, so this was only seen in the second last layer.
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class BasicModel(nn.Module):
    # Wrapper for a model, containing the train and test functions
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        # Clear the optimizer to 0
        self.optimizer.zero_grad()
        # Forward pass through the model
        output = self(input_img, input_qst)
        # Negatve log-likelihood loss (since this is a classification problem)
        loss = F.nll_loss(output, label)
        # Compute backward gradients on the NLL loss
        loss.backward()
        # Take a step using the step
        self.optimizer.step()
        # The model prediction is the value with the highest probability
        pred = output.data.max(1)[1]
        # The model output is compared to the true label data to calculate accuracy
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch):
        # Save model parameters during training for ease of resuming training
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class RN(BasicModel):
    # Class for defining a relation network
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')

        self.input_type = args.input_type
        
        if self.input_type == "pixels":
            # Create the convolutional network using the ConvInputModel() class
            # Note that we only need the CNN when we are training on image data (pixels)
            self.conv = ConvInputModel()
        
        self.relation_type = args.relation_type

        # Define the fully connected layers of the g and f networks
        # Here, there are 4 layers of the g network
        # There is 1 layer of the f network

        if self.relation_type == 'ternary':
            ##(number of filters per object+coordinate of object)*3+question vector
            self.g_fc1 = nn.Linear((NUM_OBJECTS+2)*3+18, 256)
        else:
            ##(number of filters per object+coordinate of object)*2+question vector
            self.g_fc1 = nn.Linear((NUM_OBJECTS+2)*2+18, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, RN_INPUT_SIZE, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, RN_INPUT_SIZE, 2))
        for i in range(RN_INPUT_SIZE):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        # Define the last two layers that takes the conv and MLP output
        # and runs a softmax over the possible outputs
        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, img, qst):
        
        # If input is pixels, pass through conv network
        # Else, the state description matrix is already encoded and ready to pass into the RN
        if self.input_type == "pixels":
            x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        else:
            x = deepcopy(img)

        """g"""
        # Need to flatten dimension if using CNN
        if self.input_type == "pixels":
            mb = x.size()[0]
            n_channels = x.size()[1]
            d = x.size()[2]
            # x_flat = (64 x 25 x 24)
            # 24 'objects', each with dimension/representation of size 25
            x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        else:

            mb = x.size()[0]
            d = x.size()[2]
            # x_flat = (64 x 10 x 6)
            # 64 objects, each with dimension 10
            x_flat = x.permute(0,2,1)

        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)

        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1, RN_INPUT_SIZE, 1)
        qst = torch.unsqueeze(qst, 2)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+18)
        x_i = x_i.repeat(1, RN_INPUT_SIZE, 1, 1)  # (64x25x25x26+18)
        x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+18)
        x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, RN_INPUT_SIZE, 1)  # (64x25x25x26+18)

        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x6x6x2*26+18)

        # reshape for passing through network
        if self.input_type == "pixels":
            x_ = x_full.view(mb * (d * d) * (d * d), (NUM_OBJECTS+2)*2+18)  # (64*25*25x2*26*18) = (40.000, 70)
        else:
            x_ = x_full.view(mb * 10 * 10, (NUM_OBJECTS+2)*2+18)

        # Pass the data through the g network with ReLu transformations in between layers
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        if self.input_type == "pixels":
            x_g = x_.view(mb, (d * d) * (d * d), 256)
        else:
            x_g = x_.view(mb, 10 * 10, 256)

        x_g = x_g.sum(1).squeeze()
        
        """f"""
        # Pass the output of the g network into the f network
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)

        # Pass the output of the above into the fcout
        # which converts the data into the final normalized output
        return self.fcout(x_f)


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 18, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)

