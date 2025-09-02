# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

class Myloss(nn.Module):
    def __init__(self,epsilon=1e-8):
        super(Myloss,self).__init__()
        self.epsilon = epsilon
        return
    def forward(self,input_, label):
        entropy = - label * torch.log(input_ + self.epsilon) -(1 - label) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy)/2 
    
def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ *torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def HDA_UDA(hiddens, focals, len_s, len_t, ad_net, coeff=None, myloss=Myloss()):
    focals = focals.reshape(-1)
    ad_out = ad_net(hiddens)
    ad_out = nn.Sigmoid()(ad_out)
    dc_target = torch.from_numpy(np.array([[1]] * len_s + [[0]] * len_t)).float().cuda()

    x = hiddens 
    entropy = Entropy(x)
    entropy.register_hook(grl_hook(coeff))
    entropy = torch.exp(-entropy)
    heuristic = torch.mean(torch.abs(focals))

    source_mask = torch.ones_like(entropy)
    source_mask[len_t:] = 0
    target_mask = torch.ones_like(entropy)
    target_mask[0:len_s] = 0
    return myloss(ad_out,dc_target), heuristic
