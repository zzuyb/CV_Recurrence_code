from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
class VOCDataSet(Dataset):
    """
    读取解析PASCAL VOC2007/2012数据集
    """
    def __init__(self,voc_root,year="2012",transforms=None,txt_name:str="train.txt"):
        assert year in ["2007","2012"],"year must be in ['2007', '2012']"
        self.root=os.path.join(voc_root,"VOCdevkit")