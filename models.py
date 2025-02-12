import torch.nn as nn
import torch
from typing import TypeVar
from torch.nn import functional as F
from copy import deepcopy
import numpy as np

import data_config

Tensor = TypeVar('torch.tensor')

class IMCLNet(nn.Module):
    def __init__(self, predictor):
        super(IMCLNet, self).__init__()
        self.predictor = predictor
        self.coefficient = torch.nn.Parameter(data_config.label_graph_norm_empty_diag)

    def forward(self, inputs):
        x = self.predictor(inputs)
        outputs = x



class CorrelateEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, noise_flag=False, fix_source=False):
        super(CorrelateEncoderDecoder, self).__init__()
        self.encoder = encoder
        if fix_source == True:
            for p in self.parameters():
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
        self.decoder = decoder
        self.noise_flag = noise_flag


    def forward(self, inputs: Tensor) -> Tensor:
        if self.noise_flag:
            encoded_input = self.encode(inputs + torch.randn_like(inputs, requires_grad=False)*0.1)
        else:
            encoded_input = self.encode(inputs)
        output = self.decoder(encoded_input)
        return output

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def loss_function(self, inputs, recons):
        recons_loss = F.mse_loss(inputs, recons)
        return recons_loss


class CorrelateMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, drop=0.1, act_fn=nn.SELU, **kwargs):
        super(CorrelateMLP, self).__init__()
        self.output_dim = output_dim
        self.drop = drop
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    act_fn(),
                    nn.Dropout(self.drop))
            )
        self.module = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.coefficient = torch.nn.Parameter(torch.from_numpy(data_config.label_graph_norm_empty_diag).float())

    def forward(self, inputs):
        embed = self.module(inputs)
        output = self.output_layer(embed)
        diag = torch.diag(self.coefficient)
        c_diag = torch.diag_embed(diag)
        output = output + torch.mm(output, self.coefficient - c_diag)
        return output






class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, drop=0.1, act_fn=nn.SELU, **kwargs):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.drop = drop
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    act_fn(),
                    nn.Dropout(self.drop))
            )
        self.module = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim, bias=True))

    def forward(self, inputs):
        embed = self.module(inputs)
        output = self.output_layer(embed)
        return output




class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, noise_flag=False, fix_source=False):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        if fix_source == True:
            for p in self.parameters():
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
        self.decoder = decoder
        self.noise_flag = noise_flag

    def forward(self, inputs: Tensor) -> Tensor:
        if self.noise_flag:
            encoded_input = self.encode(inputs + torch.randn_like(inputs, requires_grad=False)*0.1)
        else:
            encoded_input = self.encode(inputs)
        output = self.decoder(encoded_input)
        return output

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def loss_function(self, inputs, recons):
        recons_loss = F.mse_loss(inputs, recons)
        return recons_loss


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

def one_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=1)
        nn.init.zeros_(m.bias)

def two_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=2)
        nn.init.zeros_(m.bias)

def three_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=3)
        nn.init.zeros_(m.bias)

def four_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=4)
        nn.init.zeros_(m.bias)

def hun_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=100)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class HDA(nn.Module):
  def __init__(self, encoder, classifier, new_cls=True, hiden_dim=64, heuristic_num=1, heuristic_initial=False):
    super(HDA, self).__init__()
    self.feature_layers = encoder
    self.classifier = classifier
    self.new_cls = new_cls
    # cost from the start to the nth node
    self.heuristic_num = heuristic_num
    input_dim = encoder.output_layer[0].out_features
    if new_cls:
        self.fc = nn.Linear(input_dim, hiden_dim)
        if heuristic_initial:
            self.fc.apply(hun_weights)
        else:
            self.fc.apply(init_weights)
        self.heuristic = nn.Linear(input_dim, hiden_dim)
        self.heuristic.apply(init_weights)
        self.heuristic1 = nn.Linear(input_dim, hiden_dim)
        self.heuristic1.apply(one_weights)
        self.heuristic2 = nn.Linear(input_dim, hiden_dim)
        self.heuristic2.apply(two_weights)
        self.heuristic3 = nn.Linear(input_dim, hiden_dim)
        self.heuristic3.apply(three_weights)
        self.heuristic4 = nn.Linear(input_dim, hiden_dim)
        self.heuristic4.apply(four_weights)
        self.in_features = input_dim
    else:
        self.fc = nn.Linear(input_dim, hiden_dim)

  def forward(self, x, heuristic=True):
    x = self.feature_layers(x)
    if self.heuristic_num==1:
        geuristic = self.heuristic(x)
    elif self.heuristic_num==2:
        now1 = self.heuristic(x)
        now2 = self.heuristic1(x)
        geuristic = now1+now2 
    elif self.heuristic_num==3:
        now1 = self.heuristic(x)
        now2 = self.heuristic1(x)
        now3 = self.heuristic2(x)
        geuristic = (now1+now2+now3)
    elif self.heuristic_num==4:
        now1 = self.heuristic(x)
        now2 = self.heuristic1(x)
        now3 = self.heuristic2(x)
        now4 = self.heuristic3(x)
        geuristic = (now1+now2+now3+now4)
    elif self.heuristic_num==5:
        now1 = self.heuristic(x)
        now2 = self.heuristic1(x)
        now3 = self.heuristic2(x)
        now4 = self.heuristic3(x)
        now5 = self.heuristic4(x)
        geuristic = (now1+now2+now3+now4+now5)
    y = self.fc(x)
    if heuristic:
        y = y - geuristic

    # print("y=================")
    # print(y)
    out = self.classifier(y)
    # print("out===============")
    # print(out)
    return x, y, geuristic, out.squeeze(dim=1)

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
		    {"params":self.heuristic1.parameters(), "lr_mult":10, 'decay_mult':2},
		    {"params":self.heuristic2.parameters(), "lr_mult":10, 'decay_mult':2},
		    {"params":self.heuristic3.parameters(), "lr_mult":10, 'decay_mult':2},
		    {"params":self.heuristic4.parameters(), "lr_mult":10, 'decay_mult':2},
            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2},
            {"params":self.heuristic.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]  
    return parameter_list


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size = 32, out_size=1):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, out_size)
    self.relu1 = nn.ReLU()   
    self.dropout1 = nn.Dropout(0.2)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    y = self.ad_layer2(x)
    return y

  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]