import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, GELU
from utils.utils import initialize_weights
import numpy as np

from .layer import GlobalAdaptPooling, AdaptFullConv, MyTransformerConv
from torch_geometric.nn import TransformerConv, LayerNorm

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class adaptblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, init_theta=0.5, layer_num=3, edge_dim=None, return_edge=True):
        super(adaptblock, self).__init__()
        assert out_channels % heads == 0

        self.return_edge = return_edge

        self.adapt = ModuleList()
        self.adapt.append(AdaptFullConv(in_channels, in_channels//heads, heads=heads,
                                        init_theta=init_theta, edge_dim=edge_dim,
                                        use_rpe=False))
        self.adapt.append(LayerNorm(in_channels))
        self.adapt.append(GELU())

        self.layer = ModuleList()
        for i in range(layer_num):
            sublayer = ModuleList()
            # sublayer.append(AdaptFullConv(in_channels, in_channels//heads, heads=heads,
            #                               init_theta=init_theta,edge_dim=edge_dim,
            #                               use_rpe=False))
            sublayer.append(TransformerConv(in_channels, in_channels//heads,
                                            heads=heads, edge_dim=edge_dim))
            sublayer.append(LayerNorm(in_channels))
            sublayer.append(GELU())
            self.layer.append(sublayer)

        outlayer = ModuleList()
        outlayer.append(TransformerConv(in_channels, out_channels,
                                        heads=heads, edge_dim=edge_dim))
        outlayer.append(GELU())
        self.layer.append(outlayer)

    def forward(self, x, edge_index, position=None, edge_attr=None):
        edge_record = []
        beta_multi_record = []
        head_score_param = []
        residual = x

        info = self.adapt[0](x, edge_index=edge_index, position=position, edge_attr=edge_attr)
        x, new_edge_index, gate = info['out'], info['edge_index'], info['gate']
        if edge_attr is not None:
            new_edge_attr = edge_attr[gate]
        else:
            new_edge_attr = None
        x = self.adapt[1](x)
        x = self.adapt[2](x)
        if self.return_edge:
            edge_record.append(new_edge_index.cpu().detach())
        beta_multi_record.append(info['beta_multi'])
        head_score_param.append(info['param_multi'])
        x = x + residual

        # for sublayer in self.layer:
        #     residual = x
        #     info = sublayer[0](x, edge_index=new_edge_index, edge_attr=new_edge_attr)
        #     if sublayer != self.layer[-1]:
        #         x, new_edge_index, gate = info['out'], info['edge_index'], info['gate']
        #         beta_multi_record.append(info['beta_multi'])
        #         head_score_param.append(info['param_multi'])
        #         if edge_attr is not None:
        #             new_edge_attr = new_edge_attr[gate]
        #         else:
        #             new_edge_attr = None
        #         if self.return_edge:
        #             edge_record.append(new_edge_index.cpu().detach())
        #     else:
        #         x = info

        #     x = sublayer[1](x)
        #     if sublayer != self.layer[-1]:
        #         x = sublayer[2](x)
        #         x += residual
        for sublayer in self.layer:
            residual = x
            x = sublayer[0](x, edge_index=new_edge_index, edge_attr=new_edge_attr)

            x = sublayer[1](x)
            if sublayer != self.layer[-1]:
                x = sublayer[2](x)
                x += residual

        return_info = {}
        return_info['x'] = x
        return_info['out'] = x
        return_info['new_position'] = position
        return_info['new_edge_index'] = new_edge_index
        return_info['new_edge_attr'] = new_edge_attr
        return_info['beta_h_record'] = beta_multi_record
        return_info['param_h'] = head_score_param
        if self.return_edge:
            return_info['edge_record'] = edge_record
        return return_info

class transformerblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, layer_num=3, edge_dim=None):
        super(transformerblock, self).__init__()
        assert out_channels % heads == 0

        self.adapt = ModuleList()
        self.adapt.append(TransformerConv(in_channels, in_channels//heads, heads=heads, edge_dim=edge_dim))
        self.adapt.append(LayerNorm(in_channels))
        self.adapt.append(GELU())

        self.layer = ModuleList()
        for i in range(layer_num):
            sublayer = ModuleList()
            sublayer.append(TransformerConv(in_channels, in_channels//heads, heads=heads, edge_dim=edge_dim))
            sublayer.append(LayerNorm(in_channels))
            sublayer.append(GELU())
            self.layer.append(sublayer)

        outlayer = ModuleList()
        outlayer.append(TransformerConv(in_channels, out_channels, heads=heads, edge_dim=edge_dim))
        outlayer.append(GELU())
        self.layer.append(outlayer)

    def forward(self, x, edge_index, position=None, edge_attr=None):
        residual = x

        x = self.adapt[0](x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.adapt[1](x)
        x = self.adapt[2](x)
        x = x + residual

        for sublayer in self.layer:
            residual = x
            x = sublayer[0](x, edge_index=edge_index, edge_attr=edge_attr)

            x = sublayer[1](x)
            if sublayer != self.layer[-1]:
                x = sublayer[2](x)
                x += residual

        return x

class GraphAttn(nn.Module):
    def __init__(self, L=1024, D=256, heads=1, dropout=False, n_classes=1):
        super(GraphAttn, self).__init__()
        self.in_layer = TransformerConv(L, D//heads, heads=heads)
        self.act = [nn.GELU()]
        if dropout:
            self.act.apend(nn.Dropout(0.25))
        self.act = nn.Sequential(*self.act)
        self.global_pooling = GlobalAdaptPooling(D, 1, D, D, heads=n_classes, return_attention_weights=True)
    
    def forward(self, x, edge_index, label):
        x = self.in_layer(x, edge_index)
        x = self.act(x)
        y, _, attn = self.global_pooling(x, x)
        A = attn[:, label]
        return y, A, x

class GraphMultiAttn(GraphAttn):
    def __init__(self, L=1024, D=256, heads=1, layer_num=3, dropout=False, n_classes=1):
        super(GraphMultiAttn, self).__init__(L, D, heads, dropout, n_classes)
        if (D%heads!=0)|(L%heads!=0):
            raise ValueError('D or L %heads is not 0!')
        self.in_layer = transformerblock(L, D//heads, heads, layer_num=layer_num)

class AdGraphAttn(GraphAttn):
    def __init__(self, L=1024, D=256, heads=1, dropout=False, n_classes=1, init_theta=0.5):
        super(AdGraphAttn, self).__init__(L, D, heads, dropout, n_classes)
        self.in_layer = AdaptFullConv(L, D//heads, heads, init_theta=init_theta)
    
    def forward(self, x, edge_index, label):
        output = self.in_layer(x, edge_index)
        x = output['out']
        x = self.act(x)
        y, _, attn = self.global_pooling(x, x)
        # A = attn[:, label]
        allA = attn
        newoutput['x'] = output['out']
        newoutput['new_edge_index'] = output['edge_index']
        newoutput['edge_record'] = [output['edge_index']]
        newoutput['beta_h_record'] = [output['beta_multi']]
        newoutput['param_h'] = [output['param_multi']]
        return y, allA, x, newoutput

class AdGraphMultiAttn(GraphAttn):
    def __init__(self, L=1024, D=256, heads=1, layer_num=3, dropout=False, n_classes=1, init_theta=0.5):
        super(AdGraphMultiAttn, self).__init__(L, D, heads, dropout, n_classes)
        if (D%heads!=0)|(L%heads!=0):
            raise ValueError('D or L %heads is not 0!')
        self.in_layer = adaptblock(L, D//heads, heads, layer_num=layer_num, init_theta=init_theta, return_edge=True)
    
    def forward(self, x, edge_index, label):
        output = self.in_layer(x, edge_index)
        x = output['out']
        x = self.act(x)
        y, _, attn = self.global_pooling(x, x)
        # A = attn[:, label]
        allA = attn
        return y, allA, x, output
"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_Graph_SB(nn.Module):
    def __init__(self, in_channel=512, feat_channel=256, gate = True, size_arg = "small",
                 dropout = False, k_sample=8, n_classes=2, heads=1, use_multilayer=False, layer_num=1,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False,):
        super(CLAM_Graph_SB, self).__init__()
        self.size_dict = {"small": [512, feat_channel, 256], "big": [512, 512, 384]}
        size = self.size_dict[size_arg]
        # fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        # if dropout:
        #     fc.append(nn.Dropout(0.25))
        # if gate:
        #     attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        # else:
        #     attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        # fc.append(attention_net)
        # self.attention_net = nn.Sequential(*fc)
        # self.classifiers = nn.Linear(size[1], n_classes)
        if use_multilayer is False:
            self.graphClassifiers = GraphAttn(size[0], size[1], heads=heads, dropout=dropout, n_classes=n_classes)
        else:
            self.graphClassifiers = GraphMultiAttn(size[0], size[1], heads, layer_num, dropout, n_classes)
        
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graphClassifiers = self.graphClassifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        if A.shape[1] == 1:
            A = torch.transpose(A, 1, 0)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        if A.shape[1] == 1:
            A = torch.transpose(A, 1, 0)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, data, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = data.x.device
        h = data.x
        edge_index = data.edge_index
        logits, A, h = self.graphClassifiers(h, edge_index, label)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, None, results_dict

class CLAM_AdGraph_SB(CLAM_Graph_SB):
    def __init__(self, in_channel=512, feat_channel=256, gate = True, size_arg = "small",
                 dropout = False, k_sample=8, n_classes=2, heads=1, use_multilayer=False, layer_num=1,
                 init_theta=0.5, instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False
                ):
        super(CLAM_AdGraph_SB, self).__init__(in_channel, feat_channel, gate, size_arg, dropout, k_sample, n_classes, heads,
                                              use_multilayer, layer_num, instance_loss_fn, subtyping)
        size = self.size_dict[size_arg]
        if use_multilayer is False:
            self.graphClassifiers = AdGraphAttn(size[0], size[1], heads, dropout, n_classes, init_theta)
        else:
            self.graphClassifiers = AdGraphMultiAttn(size[0], size[1], heads, layer_num, dropout, n_classes, init_theta)
        # initialize_weights(self.graphClassifiers)

    def forward(self, data, label=None, instance_eval=False, return_features=False, return_attention=False):
        device = data.x.device
        h = data.x
        edge_index = data.edge_index
        logits, allA, h, output = self.graphClassifiers(h, edge_index, label)
        A = allA[:, label]

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': h})
        if return_attention:
            results_dict.update({'attention': allA})
        results_dict.update({'output': output})
        return logits, Y_prob, Y_hat, None, results_dict
