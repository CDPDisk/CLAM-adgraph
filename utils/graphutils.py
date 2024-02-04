import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import coalesce, is_undirected
from typing import Union, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

def connected_subgraph(edge_index: Tensor, size: int=None):
    if size is None:
        size = int(edge_index.max()) + 1
    seen = set()
    vertices = torch.arange(size).int().tolist()
    # G_adj = torch.sparse_coo_tensor(edge_index,
    #                                 values=torch.ones(edge_index.shape[1]), 
    #                                 size=(size,size))
    G = SparseTensor(row = edge_index[0], col=edge_index[1],
                     value=edge_index.new_ones(edge_index.shape[1]),
                     sparse_sizes = (size, size))
    G_csr = G.csr()
    # Convert the tensor to list because the calculate seed of set is much faster than tensor
    G_csr = [i.tolist() for i in G_csr]
    for v in vertices:
        if v not in seen:
            c = _plain_bfs(G_csr, v)
            seen.update(c)
            yield list(c)


# reference: https://github.com/networkx/networkx/blob/main/networkx/algorithms/components/connected.py
def _plain_bfs(G, source):
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                seen.add(v)
                nextlevel.update(G[1][G[0][v]:G[0][v+1]])
    return seen

def to_symmetry(edge_index):
    # assert is_undirected(edge_index)
    i, j = edge_index
    mask = i < j
    edge_index = torch.concat([edge_index[:, mask], edge_index[:, mask].flip(0)], axis=1)
    return edge_index

def is_symmetry(edge_index):
    n_edge = edge_index.shape[1]
    if n_edge%2!=0:
        return False
    
    i, j = edge_index
    if not (i[0:n_edge//2].equal(j[n_edge//2:]) and j[0:n_edge//2].equal(i[n_edge//2:])):
        return False

    return True

def make_connect_knn(position, k, use_stratification=False, celltype=None, node_num=None):
    if use_stratification is True:
        if celltype is None:
            raise ValueError("celltype must be provided when use_stratification is True")
    
    if node_num is None:
        node_num = len(position)
    celltype_unique = torch.unique(celltype)
    # 按细胞类型分层连接
    if use_stratification:
        _numarr = torch.arange(node_num)
        edge_index = torch.empty((2, 0)).long()
        for i in celltype_unique:
            for j in celltype_unique:
                index_fr = celltype.view(-1)==i
                index_to = celltype.view(-1)==j
                num_index_fr = _numarr[index_fr]
                num_index_to = _numarr[index_to]
                tree = cKDTree(position.numpy()[index_fr])
                if i==j:
                    _, neighbor = tree.query(position.numpy()[index_fr], k=k+1, p=2)
                    neighbor = torch.tensor(neighbor[:, 1:])
                else:
                    _, neighbor = tree.query(position.numpy()[index_to], k=k, p=2)
                neighbor = num_index_fr[neighbor.flatten()]
                knnmask = neighbor!=node_num
                temp_edge_index = torch.stack([num_index_to.repeat_interleave(k),
                                               neighbor], dim=0)[:, knnmask].long()
                edge_index = torch.cat([edge_index, temp_edge_index], dim=1)

    # 无视细胞类型全连接
    else:
        tree = cKDTree(position.numpy())
        _, neighbor = tree.query(position.numpy(), k=k+1, p=2)
        neighbor = torch.tensor(neighbor[:, 1:]).view(-1)
        knnmask = neighbor!=node_num
        edge_index = torch.stack([torch.arange(node_num).repeat_interleave(k),
                                  neighbor], dim=0)[:, knnmask].long()
    
    return edge_index
def plot_graph(edge_index, position, edge_value=None, node_value=None, save_path=None,
               use_edge_color=False, use_node_color=False,
               use_edgeannotate=False, use_nodeannotate=False,
               use_arrow=False):
    if isinstance(edge_index, Tensor):
        edge_index = edge_index.cpu().numpy()
    if isinstance(position, Tensor):
        position = position.cpu().numpy()
    if isinstance(edge_value, Tensor):
        edge_value = edge_value.cpu().numpy()
    if isinstance(node_value, Tensor):
        node_value = node_value.cpu().numpy()

    ppi = 72
    # 创建图形对象
    fig_size = 8
    node_size = 50
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=600)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

    xmin, xmax = position.min(), position.max()
    xmin = xmin - (xmax - xmin) * 0.1
    xmax = xmax + (xmax - xmin) * 0.1
    ax.set_xlim(xmin, xmax)
    ax.set_aspect('equal')
    
    ax_length=ax.bbox.get_points()[1][0]-ax.bbox.get_points()[0][0]

    # 计算节点的半径
    ax_point = ax_length*ppi/fig.dpi

    xsize=xmax-xmin
    fact=ax_point/xsize

    node_radius = (node_size**0.5)/(2*fact)
    
    edge_bias_width = node_radius

    # 根据edge_value的大小设定边的颜色和粗细
    edge_color = list()
    for i in range(edge_index.shape[1]):
        if use_edge_color and edge_value is not None:
            edge_value = (edge_value - edge_value.min()) / (edge_value.max() - edge_value.min())
            edge_color.append(plt.cm.bwr(edge_value[i]))
        else:
            edge_color.append('k')

    edge_width = edge_bias_width / 4
    
    # 绘制箭头参数
    start = position[edge_index[0, :]]
    end = position[edge_index[1, :]]
    arrow_width = edge_width * 4
    arrow_head_width = arrow_width * 2
    arrow_head_length = arrow_width * 2
    diff = end - start
    dx, dy = diff[:, 0], diff[:, 1]
    length = np.sqrt(dx**2 + dy**2)

    if use_arrow is False:
        arrow_head_width=0
        arrow_head_length=0
    # 绘制边
    for i in range(edge_index.shape[1]):
        if length[i] != 0:
            dx[i] /= length[i]
            dy[i] /= length[i]
            # ax.arrow(start[0], start[1], dx*(length-arrow_head_length), dy*(length-arrow_head_length),
            #          head_width=arrow_head_width, head_length=arrow_head_length, fc=edge_color, ec=edge_color,
            #          length_includes_head=True, lw=1, alpha=0.7)

            ax.arrow(start[i, 0]+node_radius*dx[i], start[i, 1]+node_radius*dy[i],
                     dx[i]*(length[i]-node_radius-arrow_head_length),
                     dy[i]*(length[i]-node_radius-arrow_head_length),
                     head_width=arrow_head_width,
                     head_length=arrow_head_length, fc=edge_color[i], ec=edge_color[i], length_includes_head=True,
                     lw=arrow_width, alpha=0.7)
            # ax.arrow(start[0], start[1], dx*(length-arrow_head_length), dy*(length-arrow_head_length),
            #          fc=edge_color, ec=edge_color,alpha=0.7)
    
    # 绘制边的权重
    mid = (start - end) * 0.75 + end
    if use_edgeannotate:
        for i in range(edge_index.shape[1]):
            ax.text(mid[i, 0], mid[i, 1], f'{edge_value[i].item():.2f}', ha='center', va='center', fontsize=8)
    if use_nodeannotate:
        for i in range(position.shape[0]):
            ax.text(position[i, 0], position[i, 1], f'{i}', ha='center', va='center', fontsize=8)

    if use_node_color:
        assert node_value is not None, "node_values must be provided when use_node_color is True"
        continued = np.issubdtype(node_value.dtype, float)
        if continued:
            _, node_value = np.unique(node_value, return_inverse=True)
            cmap='tab20b'
        else:
            cmap='Reds'
        scatter = ax.scatter(position[:, 0], position[:, 1], c=node_value, cmap=cmap, s=node_size, edgecolors='black')
    else:
        scatter = ax.scatter(position[:, 0], position[:, 1], s=node_size)

    legend1 = ax.legend(*scatter.legend_elements(), title="Colors")
    ax.add_artist(legend1)

    # 显示图形
    if save_path:
        plt.savefig(save_path, dpi=600)
    else:
        plt.show()

if __name__ == '__main__':
    from torch import tensor
    from torch_geometric.utils import to_undirected
    edge_index = tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3],
                         [2, 4], [3, 4], [4, 5], [5, 6], [5, 9], [6, 9], [6, 13], [7, 8],
                         [7, 10], [8, 10], [8, 11], [8, 12], [9, 13], [9, 14], [9, 16],
                         [10, 11], [10, 15], [10, 17], [11, 12], [11, 15], [12, 15], [12, 18],
                         [13, 16], [13, 19], [14, 16], [14, 20], [14, 22], [15, 17], [15, 18],
                         [16, 19], [16, 20], [16, 21], [17, 18], [19, 21], [20, 21], [20, 22],
                         [21, 22]]).T
    edge_index = tensor([[ 0,  1,  1,  2,  3,  4,  4,  4,  4,  5,  5,  8,  8,  8, 10, 10, 10, 10,
         12, 12, 12, 12, 13, 13, 14, 14, 14, 14, 15, 15, 16, 16, 16, 16, 17, 19,
         19, 20, 20, 21, 21],
        [ 1,  0,  5,  1,  1,  1,  2,  3,  5,  4,  6,  7, 11, 12,  7,  8, 15, 17,
          8, 11, 15, 18,  6, 19,  9, 16, 20, 22, 10, 18,  9, 13, 14, 21, 18, 16,
         21, 16, 21, 19, 22]])
    edge_index = to_undirected(edge_index)
    for i in connected_subgraph(edge_index):
        print(f'{i}, {len(i)}')