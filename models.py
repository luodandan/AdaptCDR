import torch.nn as nn
import torch
from typing import TypeVar
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
import config

Tensor = TypeVar('torch.tensor')

class CorrelateEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, noise_flag=False, fix_source=False):
        super(CorrelateEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if fix_source == True:
            for p in self.parameters():
                p.requires_grad = False
                print("Layer weight is freezed:",format(p.shape))
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
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    act_fn(),
                    nn.Dropout(self.drop))
            )
        self.module = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.coefficient = torch.nn.Parameter(torch.from_numpy(config.label_graph_diag).float())

    def forward(self, inputs):
        embed = self.module(inputs)
        output = self.output_layer(embed)
        
        diag = torch.diag(self.coefficient)
        c_diag = torch.diag_embed(diag)
        output = output + torch.mm(output, self.coefficient - c_diag)
        return output

class LabelDependencySmoothing(nn.Module):
    def __init__(self, label_graph, k_alpha=10, lambda_smooth=0.1, device='cuda'):
        super().__init__()
        self.k_alpha = k_alpha
        self.lambda_smooth = lambda_smooth
        self.num_labels = label_graph.shape[0]
        
        # 构建稀疏邻接矩阵
        self.adj_matrix = self._build_adjacency_matrix(label_graph)
        self.adj_matrix = torch.tensor(self.adj_matrix, dtype=torch.float).to(device)
        
        # 可学习的连接权重
        self.edge_weights = nn.Parameter(torch.ones(self.adj_matrix.nonzero().size(0)))
        
    def _build_adjacency_matrix(self, label_graph):
        """构建k-最近邻的邻接矩阵"""
        adj_matrix = np.zeros((self.num_labels, self.num_labels))
        
        for i in range(self.num_labels):
            # 获取当前标签最相关的k_alpha个标签
            # 排除自身 (co-occurrence rate=1)
            all_indices = np.arange(len(label_graph[i]))
            # 排除自身索引
            other_indices = all_indices[all_indices != i]
            # 选择相关性最高的k_alpha个标签
            neighbors = other_indices[np.argsort(label_graph[i][other_indices])[-self.k_alpha:]]
            
            for j in neighbors:
                adj_matrix[i, j] = 1.0  # 创建无向边
                adj_matrix[j, i] = 1.0
        
        return adj_matrix

    def forward(self, logits, labels):
        """
        修正：根据图片中的公式(3)正确使用labels参数
        y_{j,i} = { 
            y_{j,i}^o,  if j ∈ Ω_i (有标注标签)
            2p_{j,i}-1, if j ∉ Ω_i (无标注标签)
        }
        """
        # 1. 使用sigmoid获取预测概率
        y_prob = torch.sigmoid(logits)
        
        # 2. 根据公式(3)构建y_converted:
        #   - 有标注标签: 使用真实标签
        #   - 无标注标签: 使用预测概率
        # 这里假设labels值域为[0,1]，将其变换为[-1,1]区间
        y_label_transformed = 2 * labels - 1
        
        # 3. 区分有标注标签和预测标签
        # 在实际应用中，可以通过mask或有效标签标记，但根据图片描述
        # 我们假设标注可见性通过Ω_i隐式表示在标签值中
        # 这里简单假设: 当label不是0或1时，表示该标签无标注
        has_annotation = (labels == 0) | (labels == 1)
        y_converted = torch.where(has_annotation, 
                                 y_label_transformed, 
                                 2 * y_prob - 1)
        
        # 4. 获取邻接矩阵中的边索引
        rows, cols = self.adj_matrix.nonzero(as_tuple=False).T
        edge_ids = torch.stack([rows, cols], dim=1)
        
        # 5. 提取连接的标签对
        left_labels = edge_ids[:, 0]
        right_labels = edge_ids[:, 1]
        
        # 6. 计算标签对之间的差异
        y_left = y_converted[:, left_labels] 
        y_right = y_converted[:, right_labels] 
        diff = y_left - y_right
        
        # 7. 加权平滑损失计算
        edge_loss = self.edge_weights.unsqueeze(0) * (diff ** 2)  # (batch_size, num_edges)
        total_loss = torch.mean(edge_loss)
        
        return self.lambda_smooth * total_loss
    
    
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
    return np.float32(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

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

    out = self.classifier(y)

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