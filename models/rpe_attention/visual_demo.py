import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'

#%% 绘制embedding类型
def visual_rpe(pos_k, rp_bucket, num_buckets=None):
        
    if num_buckets is None:
        num_buckets=rp_bucket.max()+1
    lookup_table_weight = torch.randperm(num_buckets)
    max_x = pos_k[:, 0].max() - pos_k[:, 0].min() + 1
    max_y = pos_k[:, 1].max() - pos_k[:, 1].min() + 1

    fig, ax = plt.subplots()
    if pos_k.dtype.is_signed:
        assert max_x*max_y == len(pos_k)
        ax.pcolor(lookup_table_weight[rp_bucket].reshape(max_x, max_y), edgecolors='k', cmap='tab20')
    elif pos_k.dtype == torch.float:
        raise NotImplementedError('')
    # ax.set_xlim([0, 20])
    # ax.set_ylim([0, 20])

    # 设置坐标轴的位置和不可见
    ax.spines['left'].set_position(('data', max_x))
    ax.spines['right'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_position(('data', max_y))

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 删除 x 轴刻度线和标签
    ax.set_xticks([])
    ax.set_xticklabels([])

    # 删除 y 轴刻度线和标签
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.axis('equal')
    # fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()