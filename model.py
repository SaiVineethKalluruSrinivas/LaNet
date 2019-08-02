import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.utils import shuffle
import numpy as np

class LaneNet(nn.Module):
    """
        Initiates a pytorch class for Proposed LaneNet architecture
        Input_dim      : d
        condensing_dim : D
        seq_length     : n
        hidden_dim     : H
        batch_size     : batch_size 
        output_dim     : num_classes(a.k.a num_lanes)
        num_layers     : 2
    """
    def __init__(self, input_dim, condensing_dim, seq_length, hidden_dim, batch_size, output_dim, num_layers = 1, debug=False):
        super(LaneNet, self).__init__()
        self.input_dim = input_dim 
        self.condensing_dim = condensing_dim 
        self.seq_length = seq_length 
        self.hidden_dim = hidden_dim 
        self.batch_size = batch_size 
        self.num_layers = num_layers 
        self.output_dim = output_dim 
        # Condensing layers are non-trainable avgpool1d layers to perform down-sampling.
        # replaces traditional embedding layers in LSTM archs.    
        self.condensing = nn.Sequential( 
            nn.AvgPool1d(kernel_size = 500, stride = 50)
        )

        self.lstm = nn.LSTM(input_size = self.condensing_dim, 
                            hidden_size = self.hidden_dim, 
                            num_layers = self.num_layers, 
                            batch_first = True)
        
        self.fcs = nn.ModuleList([nn.Linear(self.hidden_dim, self.output_dim) for i in range(self.seq_length)])

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim), requires_grad = True))
    
    def forward(self, x):
        self.debug = False
        if self.debug:
            print(x.shape)
        x = x.permute(1,0,2) 
        if self.debug:
            print(x.shape)
        y = torch.randn(1, self.batch_size, self.condensing_dim).type(torch.FloatTensor)
        for i in range(self.seq_length):
            if self.debug:
                plt.plot(x[i].view(self.batch_size, 1, self.input_dim)[0][0].data.numpy())
            curr_seq_condensing = self.condensing(x[i].view(self.batch_size, 1, self.input_dim))
            if self.debug:
                print(curr_seq_condensing.shape)
                plt.plot(curr_seq_condensing[0][0].data.numpy())
            curr_seq_condensing = curr_seq_condensing.view(1,self.batch_size, -1) 
            y = torch.cat((y,curr_seq_condensing), dim = 0)
        
        if self.debug:
            print(y.shape)
        y = y[1:].permute(1,0,2)
        y = y.view(self.batch_size, self.seq_length, -1)
        if self.debug:
            print(y.shape)           
        lstm_out, self.hidden = self.lstm(y)
        return [self.fcs[i](lstm_out[:, i].view(self.batch_size, -1)) for i in range(self.seq_length)]
    
    

    

    
    