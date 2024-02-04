import torch
from torch import tensor
from torch.utils.data import Dataset

from pathlib import Path
import re
import pandas as pd
import numpy as np

HER2class = {'TNBC':tensor([0]).long(),
           'HRpHER2n':tensor([0]).long(),
           'HRnHER2p':tensor([1]).long(),
           'HRpHER2p':tensor([1]).long()}
HRclass = {'TNBC':tensor([0]).long(),
           'HRpHER2n':tensor([1]).long(),
           'HRnHER2p':tensor([0]).long(),
           'HRpHER2p':tensor([1]).long()}
resample_ratio = {'HRnHER2p':6,
                   'HRpHER2n':1,
                   'HRpHER2p':5,
                   'TNBC':4}

class PatchGraphMIL(Dataset):
    def __init__(self, root, infoDir, mode='train', resample=False):
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        infoDir = Path('D:/FUSCC/adaptgraph/otherInfo')

        fileList = [*root.glob('*_PatchGraphFeat.pth')]
        sampleRegex = re.compile(r'.+\\(?P<sample>.+)_PatchGraphFeat.pth')
        sampleList = [sampleRegex.search(str(i)).group(1) for i in fileList]

        clinicalInfo = pd.read_excel(infoDir/'clinical_info.xlsx')
        clinicalInfo = clinicalInfo[['PatientCode', 'IHC_Subtype']]
        clinicalInfo = clinicalInfo.rename(columns={'PatientCode':'sampleName', 'IHC_Subtype':'subtype'})
        clinicalInfo.index = clinicalInfo['sampleName']
        self.clinicalInfo = clinicalInfo

        sampleUsed = torch.load(infoDir/(mode+'_split.pth'))
        sampleList = sorted(list(set(sampleList)&set(clinicalInfo['sampleName'])&set(sampleUsed)))
        if resample:
            resampleList = []
            for sample in sampleList:
                resampleList += [sample]*resample_ratio[clinicalInfo.loc[sample]['subtype']]
            sampleList = resampleList

        fileList = [root/(i+'_PatchGraphFeat.pth') for i in sampleList]
        randindex = torch.randperm(len(fileList))
        fileList = np.array(fileList)[randindex].tolist()
        sampleList = np.array(sampleList)[randindex].tolist()

        self.fileList = fileList
        self.sampleList = sampleList
        
    def __len__(self):
        return len(self.fileList)
    
    def __getitem__(self, index):
        data = torch.load(self.fileList[index])
        label = Subtype2class[self.clinicalInfo.loc[self.sampleList[index], 'subtype']]
        return data, label

class PatchVisMIL(Dataset):
    def __init__(self, root, infoDir, mode='train', resample=False):
        if isinstance(root, str):
            root = Path(root)
        if isinstance(infoDir, str):
            infoDir= Path(infoDir)
        self.root = root

        fileList = [*root.glob('*_Feature.pth')]
        sampleRegex = re.compile(r'.+\\(?P<sample>.+)_Feature.pth')
        sampleList = [sampleRegex.search(str(i)).group(1) for i in fileList]

        clinicalInfo = pd.read_excel(infoDir/'clinical_info.xlsx')
        clinicalInfo = clinicalInfo[['PatientCode', 'IHC_Subtype']]
        clinicalInfo = clinicalInfo.rename(columns={'PatientCode':'sampleName', 'IHC_Subtype':'subtype'})
        clinicalInfo.index = clinicalInfo['sampleName']
        self.clinicalInfo = clinicalInfo

        sampleUsed = torch.load(infoDir/(mode+'_split.pth'))
        sampleList = sorted(list(set(sampleList)&set(clinicalInfo['sampleName'])&set(sampleUsed)))
        if resample:
            resampleList = []
            for sample in sampleList:
                resampleList += [sample]*resample_ratio[clinicalInfo.loc[sample]['subtype']]
            sampleList = resampleList

        dataList = [root/(i+'_Feature.pth') for i in sampleList]
        randindex = torch.randperm(len(dataList))
        dataList = np.array(dataList)[randindex].tolist()
        sampleList = np.array(sampleList)[randindex].tolist()

        self.dataList = dataList
        self.sampleList = sampleList
        
    def __len__(self):
        return len(self.dataList)
    
    def __getitem__(self, index):
        data = torch.load(self.dataList[index])
        label = Subtype2class[self.clinicalInfo.loc[self.sampleList[index], 'subtype']]
        return data, label

class MILGraph(Dataset):
    def __init__(self, root, infoDir, Subtype2class, mode='train', resample=False):
        if isinstance(root, str):
            root = Path(root)
        if isinstance(infoDir, str):
            infoDir = Path(infoDir)
        self.root = root
        self.Subtype2class = Subtype2class

        fileList = [*root.rglob('*_MILGraph.pth')]
        sampleRegex = re.compile('.+/(?P<sample>.+)_MILGraph.pth')
        sampleList = [sampleRegex.search(str(i)).group(1) for i in fileList]

        clinicalInfo = pd.read_excel(infoDir/'clinical_info.xlsx')
        clinicalInfo = clinicalInfo[['PatientCode', 'IHC_Subtype']]
        clinicalInfo = clinicalInfo.rename(columns={'PatientCode':'sampleName', 'IHC_Subtype':'subtype'})
        clinicalInfo.index = clinicalInfo['sampleName']
        self.clinicalInfo = clinicalInfo

        sampleUsed = torch.load(infoDir/(mode+'_split.pth'))
        sampleList = sorted(list(set(sampleList)&set(clinicalInfo['sampleName'])&set(sampleUsed)))
        if resample:
            resampleList = []
            for sample in sampleList:
                resampleList += [sample]*resample_ratio[clinicalInfo.loc[sample]['subtype']]
            sampleList = resampleList

        fileList = [root/i/(i+'_MILGraph.pth') for i in sampleList]
        randindex = torch.randperm(len(fileList))
        fileList = np.array(fileList)[randindex].tolist()
        sampleList = np.array(sampleList)[randindex].tolist()

        self.fileList = fileList
        self.sampleList = sampleList
        
    def __len__(self):
        return len(self.fileList)
    
    def __getitem__(self, index):
        Subtype2class = self.Subtype2class

        data = torch.load(self.fileList[index])
        label = Subtype2class[self.clinicalInfo.loc[self.sampleList[index], 'subtype']]
        return data, label

def dataset_split(root, infoDir,subtypeLabel, dataset=MILGraph, fold=0):
    if subtypeLabel == 'HR':
        Subtype2class = HRclass
    elif subtypeLabel == 'HER2':
        Subtype2class = HER2class
    else:
        raise ValueError
    train_dataset = dataset(root, infoDir, Subtype2class, mode='train', resample=True)
    val_dataset = dataset(root, infoDir, Subtype2class, mode='val', resample=False)
    test_dataset = dataset(root, infoDir, Subtype2class, mode='test', resample=False)
    return train_dataset, val_dataset, test_dataset

if __name__=='__main__':
    # root = 'D:/FUSCC/adaptgraph/patchGraphFeat'
    # root = 'D:/FUSCC/deep/resnet_patch224_mag40/feats'
    root = 'D:/FUSCC/adaptgraph/MILGraph'
    infoDir = Path('D:/FUSCC/adaptgraph/otherInfo')
    # mydataset = PatchVisMIL(root, infoDir, mode='train', resample=True)
    trainset, valset, testset = dataset_split(root, infoDir, dataset=MILGraph)
