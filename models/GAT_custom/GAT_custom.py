# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from torch.nn import LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, BatchNorm
from models.GAT_custom.Modified_GAT import GATConv as GATConv
from torch_geometric.nn import GraphSizeNorm

from models.GAT_custom.model_utils import weight_init
from models.GAT_custom.model_utils import decide_loss_type

from models.GAT_custom.pre_layer import preprocess
from models.GAT_custom.post_layer import postprocess

class GAT_module(torch.nn.Module):

    def __init__(self, input_dim, output_dim, head_num, dropedge_rate, graph_dropout_rate, loss_type, with_edge, simple_distance, norm_type):
        """
        :param input_dim: Input dimension for GAT
        :param output_dim: Output dimension for GAT
        :param head_num: number of heads for GAT
        :param dropedge_rate: Attention-level dropout rate
        :param graph_dropout_rate: Node/Edge feature drop rate
        :param loss_type: Choose the loss type
        :param with_edge: Include the edge feature or not
        :param simple_distance: Simple multiplication of edge feature or not
        :param norm_type: Normalization method
        """

        super(GAT_module, self).__init__()
        self.conv = GATConv([input_dim, input_dim], output_dim, heads=head_num, dropout=dropedge_rate, with_edge=with_edge, simple_distance=simple_distance)
        self.norm_type = norm_type
        if norm_type == "layer":
            self.bn = LayerNorm(output_dim * int(self.conv.heads))
            self.gbn = None
        else:
            self.bn = BatchNorm(output_dim * int(self.conv.heads))
            self.gbn = GraphSizeNorm()
        self.prelu = decide_loss_type(loss_type, output_dim * int(self.conv.heads))
        self.dropout_rate = graph_dropout_rate
        self.with_edge = with_edge

    def reset_parameters(self):

        self.conv.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_attr, edge_index, batch):

        if self.training:
            drop_node_mask = x.new_full((x.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_node_mask = torch.bernoulli(drop_node_mask)
            drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
            drop_node_feature = x * drop_node_mask

            if edge_attr is not None:
                drop_edge_mask = edge_attr.new_full((edge_attr.size(1),), 1 - self.dropout_rate, dtype=torch.float)
                drop_edge_mask = torch.bernoulli(drop_edge_mask)
                drop_edge_mask = torch.reshape(drop_edge_mask, (1, drop_edge_mask.shape[0]))
                drop_edge_attr = edge_attr * drop_edge_mask
        else:
            drop_node_feature = x
            drop_edge_attr = edge_attr

        if self.with_edge == "Y":
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=drop_edge_attr, return_attention_weights=True)
        else:
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=None, return_attention_weights=True)
        out_x_temp = 0
        if self.norm_type == "layer":
            for c, item in enumerate(torch.unique(batch)):
                temp = self.bn(x_before[batch == item])
                if c == 0:
                    out_x_temp = temp
                else:
                    out_x_temp = torch.cat((out_x_temp, temp), 0)
        else:
            temp = self.gbn(self.bn(x_before), batch)
            out_x_temp = temp

        x_after = self.prelu(out_x_temp)

        return x_after, attention_value

class GAT(torch.nn.Module):

    def __init__(self, feat_channel, heads, n_classes, with_distance=False, simple_distance='Y', layer_num=1,
                 residual_connection='Y',
                 norm_type='batch',
                 dropout=0., graph_dropout_rate=0., dropedge_rate=0., **kwargs):
        super(GAT, self).__init__()

        dim = feat_channel
        self.dropout_rate = dropout
        self.dropedge_rate = dropedge_rate
        self.heads = heads
        self.include_edge_feature = with_distance
        self.layer_num = layer_num
        self.graph_dropout_rate = graph_dropout_rate
        self.residual = residual_connection
        self.norm_type = norm_type
        MLP_layernum=0
        loss_type = 'Leaky'

        postNum = 0
        self.preprocess = preprocess(dim, MLP_layernum, dropout, norm_type, simple_distance)
        self.conv_list = nn.ModuleList([GAT_module(dim, dim//heads, heads, dropedge_rate,
                                                   graph_dropout_rate, loss_type,
                                                   with_edge=with_distance,
                                                   simple_distance=simple_distance,
                                                   norm_type=norm_type) for _ in range(layer_num)])
        postNum += int(self.heads) * len(self.conv_list)

        self.postprocess = postprocess(dim, layer_num, dim, MLP_layernum, dropout)
        self.risk_prediction_layer = nn.Linear(self.postprocess.postlayernum[-1], n_classes)

    def reset_parameters(self):

        self.preprocess.reset_parameters()
        for i in range(int(self.Argument.number_of_layers)):
            self.conv_list[i].reset_parameters()
        self.postprocess.reset_parameters()
        self.risk_prediction_layer.apply(weight_init)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)

    def forward(self, data, edge_mask=None):

        preprocessed_input, preprocess_edge_attr = self.preprocess(data, edge_mask)
        batch = data.batch

        x0_glob = global_mean_pool(preprocessed_input, batch)
        x_concat = x0_glob

        x_out = preprocessed_input
        final_x = x_out
        count = 0
        attention_list = []

        for i in range(int(self.layer_num)):
            select_idx = int(i)
            x_temp_out, attention_value = \
                self.conv_list[select_idx](x_out, preprocess_edge_attr, data.edge_index, batch)
            _, attention_value = attention_value
            if len(attention_list) == 0:
                attention_list = torch.reshape(attention_value, (1, attention_value.shape[0], attention_value.shape[1]))
            else:
                attention_list = torch.cat((attention_list, torch.reshape(attention_value, (
                1, attention_value.shape[0], attention_value.shape[1]))), 0)

            x_glob = global_mean_pool(x_temp_out, batch)
            x_concat = torch.cat((x_concat, x_glob), 1)

            if self.residual == "Y":
                x_out = x_temp_out + x_out
            else:
                x_out = x_temp_out

            final_x = x_out
            count = count + 1

        postprocessed_output = self.postprocess(x_concat, data.batch)
        risk = self.risk_prediction_layer(postprocessed_output)
        Y_prob = F.softmax(risk, dim=1)
        Y_hat = torch.topk(risk, 1, dim=1)[1]

        return risk, Y_prob, Y_hat, None, None
