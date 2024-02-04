from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.my_utils import dataset_split
from models.model_clamGraph import CLAM_Graph_SB, CLAM_AdGraph_SB

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import time


def main(args):
    train_dataset, val_dataset, test_dataset = dataset_split(args.root, args.infoDir, args.subtypeLabel, fold=0)
    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb', 'clam_graph', 'clam_adgraph']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.model_type in ['clam_graph', 'clam_adgraph']:
            model_dict.update({'heads': args.head_num})
            model_dict.update({'in_channel': args.in_channel})
            model_dict.update({'feat_channel': args.feat_channel})
            model_dict.update({'use_multilayer': args.use_multilayer})
            model_dict.update({'layer_num': args.layer_num})

    with torch.no_grad():
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data, label = train_dataset[0]
        data = data.to(device)
        label = label.to(device)

        model = CLAM_AdGraph_SB(**model_dict, init_theta=args.init_theta)
        model.relocate()
        # checkpoint = torch.load('./results/AdGraphHER128H4B20InitT0Multilayer2_s1/s_0_checkpoint.pt')
        # model.load_state_dict(checkpoint)

        starttime = time.time()
        for epoch in range(1):
            for data, label in train_dataset:
                data = data.to(device)
                label = label.to(device)
                model(data)
        endtime = time.time()
        print('Time cost of adaptgraph: ', (endtime - starttime)/1)
        print('Memory cost of adaptgraph: ', torch.cuda.memory_allocated())
        torch.cuda.empty_cache()

        model = CLAM_Graph_SB(**model_dict)
        model.relocate()
        starttime = time.time()
        for epoch in range(1):
            for data, label in train_dataset:
                data = data.to(device)
                label = label.to(device)
                model(data)
        endtime = time.time()
        print('Time cost of transformer: ', (endtime - starttime)/1)
        print('Memory cost of transformer: ', torch.cuda.memory_allocated())
        torch.cuda.empty_cache()
    print('Done!')

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
# model and size
parser.add_argument('--seed', type=int, default=2, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--drop_out', type=float, default=0., help='enable dropout')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'transmil', 'clam_graph', 'clam_adgraph'], default='clam_graph', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--head_num', type=int, default=4, help='the number of head when use graph feature extractor')
parser.add_argument('--use_multilayer', action='store_true', default=True, help='use multi layer model')
parser.add_argument('--layer_num', default=4, type=int, help='the number of layer when use multilayer')
parser.add_argument('--in_channel', type=int, default=512, help='the channel of input')
parser.add_argument('--feat_channel', type=int, default=128, help='the channel of feat')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--init_theta', type=float, default=0.5, help='init theta')

### CLAM specific options
parser.add_argument('--subtyping', action='store_true', default=True, 
                     help='subtyping problem')
parser.add_argument('--subtypeLabel', choices=['HR', 'HER2'], default='HER2', help='The subtype label used')

parser.add_argument('--root', type=str, default='/data/aim_nuist/aim_chendp/data/MILGraph/processed')
parser.add_argument('--infoDir', type=str, default='/data/aim_nuist/aim_chendp/data/MILGraph/raw')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.n_classes=2

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
