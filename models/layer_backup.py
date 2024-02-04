import os
import sys
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_sparse import SparseTensor
from torch_scatter import scatter_max, scatter_min

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as scatter_softmax

from .rpe_attention import grpe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.graphutils import connected_subgraph, is_symmetry, plot_graph, to_symmetry
from torch_geometric.utils import to_undirected

class AdaptFullConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        init_theta = 0.5,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        use_rpe: bool = False,
        rpe_config = None,
        ratio:float = 1.,
        merge_heads: bool = False,
        edge_attr_addK = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'mean')
        super(AdaptFullConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None
        self.merge_heads = concat and merge_heads
        self.edge_attr_addK = edge_attr_addK

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        self.register_parameter('theta', nn.Parameter(init_theta * torch.ones(heads, dtype=torch.float)))

        self.theta_score = nn.ReLU()
        self.head_score = nn.Sequential(nn.Linear(heads, heads, bias=False),
                                        nn.GELU(),
                                        nn.Linear(heads, 1, bias=False),
                                        nn.Sigmoid())
        # self.register_parameter('head_weight', nn.Parameter(torch.ones(heads, dtype=torch.float)))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
            if self.merge_heads:
                self.act_merge_heads = nn.GELU()
                self.lin_merge_heads = Linear(heads * out_channels, out_channels)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
            self.act_merge_heads = self.register_parameter('act_merge_heads', None)
            self.lin_merge_heads = self.register_parameter('lin_merge_heads', None)
        
        self.use_rpe = use_rpe
        if use_rpe:
            if rpe_config is None:
                rpe_config = grpe.get_rpe_config(ratio=ratio, method='mlp',
                                                 mode='contextual', shared_head=False,
                                                 rpe_on='qk')
            self.rpe_q, self.rpe_k, self.rpe_v = grpe.build_rpe(rpe_config, head_dim=out_channels, num_heads=heads)
        else:
            self.rpe_q, self.rpe_k, self.rpe_v = grpe.build_rpe(None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()

        nn.init.uniform_(self.head_score[0].weight, a=0, b=1)
        nn.init.uniform_(self.head_score[2].weight, a=0, b=1)

        if self.beta:
            self.lin_beta.reset_parameters()
        if self.merge_heads:
            self.lin_merge_heads.reset_parameters()
        for c in 'qkv':
            name = 'rpe_' + c
            rpe = getattr(self, name)
            if rpe:
                rpe.reset_parameters()

    @property
    def sigtheta(self):
        return self.theta.sigmoid()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Tensor,
                position: OptTensor = None, edge_attr: OptTensor = None,
                return_attention_weights: bool = False, visual_name: str = None):
        # # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor # noqa
        # # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa

        if not is_symmetry(edge_index):
            raise ValueError("edge_index must be symmetry")

        if self.use_rpe:
            if position is None:
                raise ValueError("Coor can't both be None when use rpe")
        
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None, pos=position, vpos=position,
                             visual_name=visual_name)

        alpha = self._alpha
        edge_index = self._edge_index
        edge_attr = self._edge_attr
        gate = self._gate
        beta_multi = self._beta_multi
        # if self.use_rpe:
        #     position = position[gate]
        self._alpha, self._edge_index, self._edge_attr, self._gate = None, None, None, None
        

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r
        
        if self.concat and self.merge_heads:
            out = self.act_merge_heads(out)
            out = self.lin_merge_heads(out)

        out_info = {}
        out_info['out'] = out
        out_info['gate'] = gate
        out_info['position'] = position
        out_info['beta_multi'] = beta_multi
        out_info['param_multi'] = list(self.head_score.parameters())
        out_info['edge_attr'] = edge_attr
                
        if return_attention_weights is True:
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                out_info['edge_index'] = edge_index
                out_info['alpha'] = alpha
            elif isinstance(edge_index, SparseTensor):
                out_info['edge_index'] = edge_index.set_value(alpha, layout='coo')
        else:
            out_info['edge_index'] = edge_index
        
        return out_info

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_index: Tensor, 
                pos_i:Tensor, pos_j:Tensor, vpos:Tensor,
                edge_attr: OptTensor,
                index: Tensor, size_i: Optional[int],
                visual_name: str) -> Tensor:

        # Compute attention coefficients.
        n_edge = edge_index.shape[1]
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            # add edge_attr to key is not necessary
            if self.edge_attr_addK:
                key_j = key_j + edge_attr
        # alpha shape: (L, H, C)
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)

        if self.use_rpe:
            # If don't provide rp_bucket, calculate it from coordinate
            pos = (pos_i, pos_j)
            rp_bucket = None
            if rp_bucket is None and pos is not None:
                for c in 'qkv':
                    name = 'rpe_' + c
                    rpe = getattr(self, name)
                    if rpe and rpe.method != grpe.METHOD.MLP:
                        rp_bucket = rpe._get_rp_bucket(pos=pos)
            
            if self.rpe_q:
                alpha += self.rpe_q(query_i, rp_bucket=rp_bucket, pos=pos)
            if self.rpe_k:
                alpha += self.rpe_k(key_j, rp_bucket=rp_bucket, pos=pos)
            if self.rpe_v:
                raise NotImplementedError
        
        # make the alpha of the same edge to be the same throught mean
        alpha = (alpha[0:n_edge//2, ...] + alpha[n_edge//2:, ...]) / 2
        alpha = torch.concat([alpha, alpha], dim=0)

        # Calculate the beta coefficient of pre-take edge
        gate = alpha.sigmoid()
        # multi_head_index = index + torch.arange(self.heads).expand(index.shape[0], -1).to(index.device) * size_i
    
        # 使用softmax注意力
        # beta_theta = alpha.new_zeros((alpha.shape[0], self.heads))
        # beta_theta = ((self.sigtheta - gate + 1e-6) / (gate - 1 - 1e-6) + 1).log()
        beta_theta = ((gate+1e-6)/(self.sigtheta+1e-6)).log()
        # beta_theta = F.relu(beta_theta)

        alpha += beta_theta
        beta_multi = self.head_score(alpha.sigmoid())+1e-6
        alpha += beta_multi.log()

        if alpha.isnan().any():
            raise ValueError
        alpha = scatter_softmax(alpha, index=index, num_nodes=size_i, dim=0)
        # # 使用sigmoid注意力
        # beta = alpha.new_ones((alpha.shape[0], self.heads))
        # beta[drop_index] = ((1 - self.sigtheta) / (1 - gate + 1e-10))[drop_index]
        # alpha = alpha.sigmoid() * beta
    
        if visual_name:
            take_index = (gate >= self.sigtheta)
            drop_index = ~take_index
            plot_graph(edge_index, vpos, edge_value=alpha,
                    save_path=f'./fig/{visual_name}_T{self.sigtheta.item():.2f}_0before.jpg',
                    use_edgeannotate=True, use_nodeannotate=True)
            plot_graph(edge_index[:, take_index], vpos, edge_value=alpha[take_index],
                    save_path=f'./fig/{visual_name}_T{self.sigtheta.item():.2f}_1drop.jpg',
                    use_edgeannotate=True, use_nodeannotate=True)

        self._alpha = alpha
        self._beta_multi = beta_multi
        with torch.no_grad():
            self._gate = beta_multi.squeeze()>self.head_score(self.sigtheta)
        self._edge_index = edge_index[:, self._gate]
        if edge_attr is not None:
            self._edge_attr = edge_attr[self._gate]
        else:
            self._edge_attr = None

        # alpha = softmax(alpha, index=index, num_nodes=size_i)

        # alpha.index_put(drop_index,
        #                 alpha[drop_index] + (beta.unsqueeze(dim=1) + 1e-6))

        if self.dropout:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class MyTransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        class_in_channels: int,
        class_out_channels:int, 
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        alpha_before: bool = False,
        edge_attr_addK = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(MyTransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.class_out_channels = class_out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.alpha_before = alpha_before

        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None
        self.edge_attr_addK = edge_attr_addK

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(class_in_channels, heads * class_out_channels)
        self.lin_query = Linear(class_in_channels, heads * class_out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        self.lin_class = Linear(class_in_channels, class_out_channels)
        self.act_class = nn.GELU()

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
            # self.lin_edgekey = Linear(edge_dim, heads * class_out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], class_token: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels
        C_class = self.class_out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(class_token).view(-1, H, C_class) 
        key = self.lin_key(class_token).view(-1, H, C_class)
        # query = class_token.view(-1, 1, C_class)
        # key = class_token.view(-1, 1, C_class)
        value = self.lin_value(x[0]).view(-1, H, C)
        out_class = self.act_class(self.lin_class(class_token))
        # out_class = class_token

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        if self.alpha_before:
            alpha = self._alpha_before
        else:
            alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, out_class, alpha
            elif isinstance(edge_index, SparseTensor):
                return out, out_class, edge_index.set_value(alpha, layout='coo')
        else:
            return out, out_class

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            if self.edge_attr_addK:
                key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.class_out_channels)
        self._alpha_before = alpha
        alpha = scatter_softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if self.lin_edge is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class AdaptPooling(nn.Module):
    def __init__(self,in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 heads: int,
                 dropout: float = 0.,
                 use_rpe: bool = False,
                 rpe_config = None,
                 ratio:float=1.,
                 concat: bool = True,
                 merge_heads:bool = False,
                 **kwargs):
        super(AdaptPooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self._alpha = None

        self.concat = concat
        self.merge_heads = concat and merge_heads
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        # self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        self.register_parameter('seed', nn.Parameter(torch.rand(1, heads, out_channels)))

        if concat and self.merge_heads:
                self.act_merge_heads = nn.GELU()
                self.lin_merge_heads = Linear(heads * out_channels, out_channels)
        else:
            self.act_merge_heads = self.register_parameter('act_merge_heads', None)
            self.lin_merge_heads = self.register_parameter('lin_merge_heads', None)

        self.use_rpe = use_rpe
        if use_rpe:
            if rpe_config is None:
                rpe_config = grpe.get_rpe_config(ratio=ratio, method='mlp',
                                                mode='contextual', shared_head=False,
                                                rpe_on='qk')
            self.rpe_q, self.rpe_k, self.rpe_v = grpe.build_rpe(rpe_config, head_dim=out_channels, num_heads=heads)
        else:
            self.rpe_q, self.rpe_k, self.rpe_v = grpe.build_rpe(None)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        if self.merge_heads:
            self.lin_merge_heads.reset_parameters()
        for c in 'qkv':
            name = 'rpe_' + c
            rpe = getattr(self, name)
            if rpe:
                rpe.reset_parameters()

        torch.nn.init.xavier_uniform_(self.seed)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Tensor,
                raw_edge_index: Tensor, position: Tensor):
        if self.use_rpe:
            if position is None:
                raise ValueError("Coor can't both be None when use rpe")

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        x_size = x[0].shape[0]

        query = self.seed.view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)
        
        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)

        # cluster graph
        batch = edge_index.new_zeros(x_size, dtype=torch.int64) # function scatter need type int64
        for i, subgraph_vertex in enumerate(connected_subgraph(edge_index, size=x_size)):
            batch[subgraph_vertex] = i
        batch_size = batch.max() + 1

        new_position = scatter(position, index=batch, dim=0, dim_size=batch_size, reduce='mean')
        if self.use_rpe:
            # calculate the new position of each vertices of pooling graph

            # Don't provide rp_bucket, so get it from coor
            new_position_expand = new_position[batch]
            pos = (new_position_expand, position)
            rp_bucket = None
            if rp_bucket is None and pos is not None:
                for c in 'qkv':
                    name = 'rpe_' + c
                    rpe = getattr(self, name)
                    if rpe and rpe.method != grpe.METHOD.MLP:
                        rp_bucket = rpe._get_rp_bucket(pos=pos)
            
            if self.rpe_q:
                alpha += self.rpe_q(query, rp_bucket=rp_bucket, pos=pos)
            if self.rpe_k:
                alpha += self.rpe_k(key, rp_bucket=rp_bucket, pos=pos)
            if self.rpe_v:
                raise NotImplementedError

        alpha = (query * key).sum(-1) / math.sqrt(self.out_channels)

        alpha = scatter_softmax(alpha, batch, num_nodes=batch_size)

        if self.dropout:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value

        out = out * alpha.view(-1, self.heads, 1)

        out = scatter(out, index=batch, dim=0, dim_size=batch_size, reduce='sum')

        # make edge_index for new vertices
        new_edge_index = torch.stack([batch[raw_edge_index[0]], batch[raw_edge_index[1]]], axis=1).T
        new_edge_index = to_undirected(new_edge_index, num_nodes=batch_size)
        new_edge_index = to_symmetry(new_edge_index)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.concat and self.merge_heads:
            out = self.act_merge_heads(out)
            out = self.lin_merge_heads(out)

        return out, new_edge_index, new_position

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class GlobalAdaptPooling(nn.Module):
    def __init__(self,in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 class_in_channels: int,
                 class_out_channels: int, 
                 heads: int,
                 dropout: float = 0.,
                 use_rpe: bool = False,
                 rpe_config = None,
                 ratio:float=1.,
                 concat: bool = True,
                 merge_heads:bool = False,
                 return_attention_weights=False,
                 **kwargs):
        super(GlobalAdaptPooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.class_out_channels = class_out_channels
        self.heads = heads
        self.dropout = dropout
        self._alpha = None

        self.concat = concat
        self.merge_heads = concat and merge_heads
        self.return_attention_weights=return_attention_weights
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(class_in_channels, heads * class_out_channels)
        # self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        self.register_parameter('seed', nn.Parameter(torch.rand(1, heads, class_out_channels)))

        self.lin_class = Linear(class_in_channels, class_out_channels)
        self.act_class = nn.GELU()

        if concat and self.merge_heads:
                self.act_merge_heads = nn.GELU()
                self.lin_merge_heads = Linear(heads * out_channels, out_channels)
        else:
            self.act_merge_heads = self.register_parameter('act_merge_heads', None)
            self.lin_merge_heads = self.register_parameter('lin_merge_heads', None)

        self.use_rpe = use_rpe
        if use_rpe:
            if rpe_config is None:
                rpe_config = grpe.get_rpe_config(ratio=ratio, method='mlp',
                                                 mode='contextual', shared_head=False,
                                                 rpe_on='qk')
            self.rpe_q, self.rpe_k, self.rpe_v = grpe.build_rpe(rpe_config, head_dim=out_channels, num_heads=heads)
        else:
            self.rpe_q, self.rpe_k, self.rpe_v = grpe.build_rpe(None)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        if self.merge_heads:
            self.lin_merge_heads.reset_parameters()
        for c in 'qkv':
            name = 'rpe_' + c
            rpe = getattr(self, name)
            if rpe:
                rpe.reset_parameters()

        # torch.nn.init.xavier_uniform_(self.seed)

    def forward(self, x: Union[Tensor, PairTensor], class_token: Tensor, position: OptTensor = None):
        if self.use_rpe:
            if position is None:
                raise ValueError("Coor can't both be None when use rpe")

        H, C = self.heads, self.out_channels
        C_class = self.class_out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        x_size = x[0].shape[0]

        query = self.seed.view(-1, H, C_class)
        key = self.lin_key(class_token).view(-1, H, C_class)
        # query = torch.zeros_like(class_token).view(-1, 1, C_class)
        # query[:, :, 0] = 1
        # key = class_token.view(-1, 1, C_class)
        value = self.lin_value(x[0]).view(-1, H, C)
        out_class = self.act_class(self.lin_class(class_token))
        

        if self.use_rpe:
            # Don't provide rp_bucket, so get it from coor
            new_position = position.mean(dim=0, keepdim=True)
            new_position_expand = new_position.expand(x_size, -1)
            pos = (new_position_expand, position)
            rp_bucket = None
            if rp_bucket is None and pos is not None:
                for c in 'qkv':
                    name = 'rpe_' + c
                    rpe = getattr(self, name)
                    if rpe and rpe.method != grpe.METHOD.MLP:
                        rp_bucket = rpe._get_rp_bucket(pos=pos)
            
            if self.rpe_q:
                alpha += self.rpe_q(query, rp_bucket=rp_bucket, pos=pos)
            if self.rpe_k:
                alpha += self.rpe_k(key, rp_bucket=rp_bucket, pos=pos)
            if self.rpe_v:
                raise NotImplementedError

        alpha = (query * key).sum(-1) / math.sqrt(C_class)
        alpha = torch.nn.Softmax(dim=0)(alpha)

        if self.dropout:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value
        out = out * alpha.view(-1, self.heads, 1)
        # out = out * alpha.view(-1, 1, 1)
        out = out.sum(axis=0, keepdim=True)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.concat and self.merge_heads:
            out = self.act_merge_heads(out)
            out = self.lin_merge_heads(out)

        if self.return_attention_weights:
            return out, out_class, alpha
        else:
            return out, out_class

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from dataloader.train_loader import MyOwnDataset
