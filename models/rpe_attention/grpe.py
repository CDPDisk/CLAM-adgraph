# %%
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, GELU, Linear
import torch.nn.functional as F
import warnings
RPEIndexFunction = None

@torch.no_grad()
def piecewise_index(relative_position, alpha, beta, gamma, dtype):
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha +
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        # round(x) when |x| <= alpha
        idx = idx.round().to(dtype)

    # assign the value when |x| > alpha
    idx[not_mask] = y_out
    return idx

@torch.no_grad()
def _rp_g_euclidean(diff, **kwargs):
    beta_int = int(kwargs['beta'])
    dis = diff.square().sum(1).float().sqrt().round()
    return piecewise_index(dis, **kwargs) + beta_int

@torch.no_grad()
def _rp_g_quant(diff, **kwargs):
    raise ValueError('Not Implement quant')

@torch.no_grad()
def _rp_g_cross_rows(diff, **kwargs):
    raise ValueError('Not implement cross row')

@torch.no_grad()
def _rp_g_cross_cols(diff, **kwargs):
    raise ValueError('Not implement cross col')

@torch.no_grad()
def _rp_g_product(diff, **kwargs):
    beta_int = int(kwargs['beta'])
    S = 2 * beta_int + 1
    # the output of piecewise index function is in [-beta_int, beta_int]
    r = piecewise_index(diff[:, 0], **kwargs) + \
        beta_int  # [0, 2 * beta_int]
    c = piecewise_index(diff[:, 1], **kwargs) + \
        beta_int  # [0, 2 * beta_int]
    pid = r * S + c
    return pid

class METHOD:
    """define iRPE method IDs
    We divide the implementation of CROSS into CROSS_ROWS and CROSS_COLS.

    """
    EUCLIDEAN = 0
    QUANT = 1
    PRODUCT = 3
    CROSS = 4
    CROSS_ROWS = 41
    CROSS_COLS = 42
    MLP = 5

_METHOD_FUNC = {
    METHOD.EUCLIDEAN: _rp_g_euclidean,
    METHOD.QUANT: _rp_g_quant,
    METHOD.PRODUCT: _rp_g_product,
    METHOD.CROSS_ROWS: _rp_g_cross_rows,
    METHOD.CROSS_COLS: _rp_g_cross_cols,
}

def get_num_buckets(method, alpha, beta, gamma):
    """ Get number of buckets storing relative position encoding.
    The buckets does not contain `skip` token.

    Parameters
    ----------
    method: METHOD
        The method ID of image relative position encoding.
    alpha, beta, gamma: float
        The coefficients of piecewise index function.

    Returns
    -------
    num_buckets: int
        The number of buckets storing relative position encoding.
    """
    beta_int = int(beta)
    if method == METHOD.PRODUCT:
        # IDs in [0, (2 * beta_int + 1)^2) for Product method
        num_buckets = (2 * beta_int + 1) ** 2
    else:
        # IDs in [-beta_int, beta_int] except of Product method
        num_buckets = 2 * beta_int + 1
    return num_buckets

@torch.no_grad()
def get_buckets_ids_g(method, pos, alpha, beta, gamma, dtype=torch.long):
    func = _METHOD_FUNC.get(method)
    pos_i, pos_j = pos
    diff = pos_j - pos_i
    bucket_ids = func(diff, alpha=alpha, beta=beta, gamma=gamma, dtype=dtype)

    # num_buckets = get_num_buckets(method, alpha, beta, gamma)
    # return bucket_ids, num_buckets
    return bucket_ids

class GRPE(nn.Module):
    def __init__(self, in_channels, num_heads,
                 mode, method,
                 transposed=True, num_buckets=None,
                 initializer=nn.init.kaiming_uniform_,
                 rpe_config=None):
        super().__init__()
        self.channel = in_channels
        self.num_heads = num_heads
        
        # relative position
        assert mode in ['bias', 'contextual']
        self.mode = mode

        assert method is not None, 'method should be a METHOD ID rather than None'
        self.method = method

        self.transposed = transposed
        self.num_buckets = num_buckets

        if initializer is None:
            def initializer(x): return None
        self.initializer = initializer

        self.reset_parameters()

        self.rpe_config = rpe_config
    
    @torch.no_grad()
    def reset_parameters(self):
        # initialize the parameters of iRPE
        if self.mode == 'bias':
            self.lookup_table_bias = nn.Parameter(
                torch.zeros(self.num_buckets, self.num_heads))
            self.initializer(self.lookup_table_bias)
        elif self.mode == 'contextual':
            if self.transposed:
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,
                                self.num_buckets, self.channel))
                self.initializer(self.lookup_table_weight)
            else:
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_buckets,
                                self.num_heads, self.channel))
                self.initializer(self.lookup_table_weight)
    
    def forward(self, x, rp_bucket=None, pos=None):
        if rp_bucket is None and pos is None:
            raise ValueError(
                "rp_bucket and pos can't both be None")

        if rp_bucket is None and pos is not None:
            rp_bucket = self._get_rp_bucket(pos)
        
        return self.forward_rpe(x, rp_bucket)
    
    def _get_rp_bucket(self, pos):
        config = self.rpe_config
        if RPEIndexFunction is not None and self.mode == 'contextual' and self.transposed:
            dtype = torch.int32
        else:
            dtype = torch.long
        # dtype = torch.int32
        
        rp_bucket = get_buckets_ids_g(method=self.method, pos=pos,
                                      alpha=config['alpha'], beta=config['beta'],
                                      gamma=config['gamma'], dtype=dtype)
        return rp_bucket
    
    def forward_rpe(self, x, rp_bucket):
        # x: The shape is (L1, H, C) rp_bucket: The shape is (L2, )
        # In AdaptConv, L1=L2. in Adapt Pool, L1=1
        # Output shape: (L2, H)
        L, H, C = x.shape
        assert (len(rp_bucket)==L) or (len(rp_bucket==1))
        
        if self.mode == 'bias':
            return self.lookup_table_bias[rp_bucket]
        elif self.mode == 'contextual':
            if not self.transposed:
                # "Suggest use contextual mode in non-transposed version"
                # lookup_table_weight: ()
                # (L2, H, C)@(L1, H, C) or (L2, H, C)@(1, H, C)
                weight = self.lookup_table_weight[rp_bucket]
                return (weight * x).sum(axis=2)
            else:
                x = x.transpose(0, 1)
                # (H, L, C)@(H, num_buckets, C) = (H, L, num_buckets)
                lookup_table = torch.matmul(x, self.lookup_table_weight)
                return lookup_table[:, :, rp_bucket].transpose(0, 1)
        else:
            raise ValueError

class GRPE_Cross(nn.Module):
    def __init__(self, method, **kwargs):
        super().init()
        raise NotImplementedError('Not Implemented cross method')

class GRPE_MLP(nn.Module):
    def __init__(self, in_channels, num_heads, mode, method, rpe_config, **kwargs):
        super(GRPE_MLP, self).__init__()
        self.mode = mode
        self.method = method
        self.rpe_config = rpe_config
        if self.mode == 'contextual':
            self.pos_encoder = Sequential(Linear(2, in_channels),
                                        GELU(),
                                        Linear(in_channels, num_heads*in_channels))
        else:
            self.pos_encoder = Sequential(Linear(2, in_channels),
                                          GELU(),
                                          Linear(in_channels, num_heads))
    def reset_parameters(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                m.reset_parameters()
            self.pos_encoder.apply(weights_init)

    def forward(self, x, pos, **kwargs):
        # The shape of x is (L, H, C)
        # The shape of position is (L, 2)
        L, H, C = x.shape
        pos_i, pos_j = pos
        diff = pos_j - pos_i
        embedding = self.pos_encoder(diff)
        if self.mode == 'contextual':
            embedding = embedding.view(-1, H, C)
            return (embedding * x).sum(dim=2)
        elif self.mode == 'bias':
            embedding = embedding.view(L, H)
            return embedding

def get_single_rpe_config(ratio=1.9,
                          method=METHOD.PRODUCT,
                          mode='contextual',
                          shared_head=True):
    config = dict()
    # whether to share encodings across different heads
    config['shared_head'] = shared_head
    # mode: None, bias, contextual
    assert mode in ['bias', 'contextual']
    config['mode'] = mode
    # method: None, Bias, Quant, Cross, Product
    config['method'] = method
    # the coefficients of piecewise index function
    config['alpha'] = 1 * ratio
    config['beta'] = 3 * ratio
    config['gamma'] = 8 * ratio

    # set the number of buckets
    config['num_buckets'] = get_num_buckets(method,
                                            config['alpha'],
                                            config['beta'],
                                            config['gamma'])
    # add extra bucket for `skip` token (e.g. class token)
    return config

def get_rpe_config(ratio=1.9,
                   method=METHOD.PRODUCT,
                   mode='contextual',
                   shared_head=True,
                   rpe_on='k'):
    if isinstance(method, str):
        method_mapping = dict(
            euc=METHOD.EUCLIDEAN,
            quant=METHOD.QUANT,
            cross=METHOD.CROSS,
            product=METHOD.PRODUCT,
            mlp=METHOD.MLP,
        )
        method = method_mapping[method.lower()]
    if mode == 'ctx':
        mode = 'contextual'
    config = dict()
    # relative position encoding on queries, keys and values
    kwargs = dict(
        ratio=ratio,
        method=method,
        mode=mode,
        shared_head=shared_head,
    )
    config['rpe_q'] = get_single_rpe_config(**kwargs) if 'q' in rpe_on else None
    config['rpe_k'] = get_single_rpe_config(**kwargs) if 'k' in rpe_on else None
    config['rpe_v'] = get_single_rpe_config(**kwargs) if 'v' in rpe_on else None
    return config

def build_rpe(config, head_dim=None, num_heads=None):
    if config is None:
        return None, None, None
    else:
        assert head_dim is not None and num_heads is not None

    rpes = [config['rpe_q'], config['rpe_k'], config['rpe_v']]
    transposeds = [False, False, False]

    def _build_single_rpe(rpe, transposed):
        if rpe is None:
            return None

        if rpe['method'] == METHOD.CROSS:
            rpe_cls = GRPE_Cross
        elif rpe['method'] == METHOD.MLP:
            rpe_cls = GRPE_MLP
        else:
            rpe_cls = GRPE

        if rpe['shared_head'] and (num_heads!=1):
            warnings.warn('Use shared head but the parameter num_heads is not 1.\
                The code will omit the setting of num_heads.')
        return rpe_cls(
            in_channels=head_dim,
            num_heads=1 if rpe['shared_head'] else num_heads,
            mode=rpe['mode'],
            method=rpe['method'],
            transposed=transposed,
            num_buckets=rpe['num_buckets'],
            rpe_config=rpe,
        )
    return [_build_single_rpe(rpe, transposed)
            for rpe, transposed in zip(rpes, transposeds)]

if __name__ == '__main__':
    x, y = torch.meshgrid(torch.arange(-10, 10, 1), torch.arange(-10, 10, 1), indexing='ij')
    pos = torch.stack([x, y], axis=-1).reshape(-1, 2).to(torch.float)
    diff = torch.tensor([[0, 0]]) - pos.view(1, -1, 2)
    diff = diff.reshape(-1, 2)


    # %%
    x, y = torch.meshgrid(torch.arange(-15, 16, 1), torch.arange(-15, 16, 1), indexing='ij')
    pos_q = torch.tensor([[0, 0]]).reshape(-1, 1, 2).to(torch.float)
    pos_k = torch.stack([x, y], axis=-1).reshape(1, -1, 2).to(torch.float)
    Lq = pos_q.shape[0]
    Lk = pos_k.shape[1]
    pos_q = pos_q.expand(Lq, Lk, 2).reshape(-1, 2)
    pos_k = pos_k.expand(Lq, Lk, 2).reshape(-1, 2)
    pos = (pos_q, pos_k)

    # %%测试rpe计算
    from visual_demo import visual_rpe
    config = get_rpe_config(ratio=1, method=METHOD.PRODUCT,
                            mode='contextual', shared_head=True,
                            rpe_on='qk')
    rpe = build_rpe(config, head_dim=20, num_heads=1)
    rpe_q, rpe_k, rpe_v = rpe
    rpe_q = rpe_q.cuda()

    rp_bucket = rpe_q._get_rp_bucket(pos=pos)
    # visual_rpe(pos_k=pos_k, rp_bucket=rp_bucket)

    x = torch.rand(Lk, 1, 20).cuda()
    pos = (pos_q.cuda(), pos_k.cuda())
    rpe_q(x, pos=pos)

    # %%测试rpe_nn
    # config = get_rpe_config(ratio=1, method=METHOD.NN,
    #                         mode='contextual', shared_head=False,
    #                         rpe_on='qk')
    # rpe = build_rpe(config, head_dim=20, num_heads=10)
    # rpe_q, rpe_k, rpe_v = rpe
    # rpe_q = rpe_q.cuda()
    # pos = (pos_q.cuda(), pos_k.cuda())
    # x = torch.rand(Lk, 10, 20).cuda()
    # rpe_q(x, pos)