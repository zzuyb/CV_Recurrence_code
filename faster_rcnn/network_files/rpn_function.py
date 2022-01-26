from typing import List, Optional, Dict, Tuple
from matplotlib.pyplot import grid

import torch
from torch import logit, nn, Tensor
from torch.nn import functional as F
import torchvision

from . import det_utils
from . import boxes as box_ops
from .image_list import ImageList

@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[int, int]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n
class AnchorsGenerator(nn.Module):
    __annotations__={
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str,List[torch.Tensor]]
    }
    def __init__(self,sizes=(128,256,512),aspect_ratios=(0.5,1.0,2.0)):
        super(AnchorsGenerator,self).__init__()
        
        if not isinstance(sizes[0],(list,tuple)):
            sizes=tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0],(list,tuple)):
            aspect_ratios=(aspect_ratios,)*len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes=sizes
        self.aspect_ratios=aspect_ratios
        self.cell_anchors=None
        self._cache={}

    def set_cell_anchors(self,dtype,device):
        # type: (torch.dtype,torch.device) -> None
        if self.cell_anchors is not None:
            cell_anchors=self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device==device:
                return

        cell_anchors=[
            self.generate_anchors(sizes,aspect_ratios,dtype,device)
            for sizes,aspect_ratios in zip(self.sizes,self.aspect_ratios)
        ]
        self.cell_anchors=cell_anchors

    def generate_anchors(self,scales,aspect_ratios,dtype=torch.float32,device=torch.device("cpu")):
        # type: (List[int],List[float],torch.dtype,torch.device) -> Tensor
        scales=torch.as_tensor(scales,dtype=dtype,device=device)
        aspect_ratios=torch.as_tensor(aspect_ratios,dtype=dtype,device=device)
        h_ratios=torch.sqrt(aspect_ratios)
        w_ratios=1.0/h_ratios

        ws=(w_ratios[:,None]*scales[None,:]).view(-1)
        hs=(h_ratios[:,None]*scales[None,:]).view(-1)

        base_anchors=torch.stack([-ws,-hs,ws,hs],dim=1)/2

        return base_anchors.round()

    def grid_anchors(self,grid_sizes,strides):
        # type: (List[List[int]],List[List[Tensor]]) -> List[Tensor]
        anchors=[]
        cell_anchors=self.cell_anchors
        assert cell_anchors is not None

        for size,stride,base_anchors in zip(grid_sizes,strides,cell_anchors):
            grid_height,grid_width=size
            stride_height,stride_width=strides
            device=base_anchors.device

            shifts_x=torch.arange(0,grid_width,dtype=torch.float32,device=device)*stride_width
            shifts_y=torch.arange(0,grid_height,dtype=torch.float32,device=device)

            shift_y,shift_x=torch.meshgrid(shifts_y,shifts_x)
            shift_x=shift_x.reshape(-1)
            shift_y=shift_y.reshape(-1)

            shifts=torch.stack([shift_x,shift_y,shift_x,shift_y],dim=1)

            shifts_anchor=shifts.view(-1,1,4)+base_anchors.view(1,-1,4)
            anchors.append(shifts_anchor.reshape(-1,4))
        return anchors



    def cached_grid_anchors(self,grid_sizes,strides):
        # type: (List[List[int]],List[List[Tensor]]) -> List[Tensor]

        key=str(grid_sizes)+str(strides)
        if key in self._cache:
            return self._cache
        anchors=self.grid_anchors(grid_sizes,strides)
        self._cache[key]=anchors
        return anchors

    def forward(self,image_list,feature_maps):
        # type: (ImageList,List[Tensor]) -> List[Tensor]
        grid_sizes=list([feature_map.shape[-2] for feature_map in feature_maps])
        image_size=image_list.tensors.shape[-2:]

        dtype,device=feature_maps[0].dtype,feature_maps[0].device

        strides=[[torch.tensor(image_size[1]//g[0],dtype=torch.int64,device=device),
                    torch.tensor(image_size[1]//g[1],dtype=torch.int64,device=device)] for g in grid_sizes]

        self.set_cell_anchors(dtype,device)

        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，对应每张预测特征图映射回原图的anchors坐标信息
        anchors_over_all_feature_maps=self.cached_grid_anchors(grid_sizes,strides)

        anchors=torch.jit.annotate(List[List[torch.Tensor]],[])

        for i ,(image_height,image_width) in enumerate(image_list.image_sizes):
            anchors_in_image=[]
            # 遍历每张预测特征图映射回原图的anchors坐标信息            
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors=[torch.cat(anchors_per_image) for anchors_per_image in anchors]
        self._cache.clear()
        return anchors





class RPNHead(nn.Module):
    def __init__(self,in_channels,num_anchors):
        super(RPNHead,self).__init__()
        self.conv=nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer,nn.Conv2d):
                torch.nn.init.normal_(layer.weight,std=0.01)
                torch.nn.init.constant_(layer.bias,0)
    
    def forward(self,x):
        logits=[]
        bbox_reg=[]
        for i,feature in enumerate(x):
            t=F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits,bbox_reg

class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Arguments:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }
    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # use during training
        # 计算anchors与真实bbox的iou
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1.
    
    def forward(self,
                images,             # type: ImageList
                features,           # type:Dict[str,Tensor]
                targets=None        # type: Optional[List[Dict[str,Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor],Dict[str,Tensor]]
        """
        """
        features=list(features.values)

        objectness,pred_bbox_deltas=self.head(features)

        anchors=self.anchor_generator(images,features)

        num_images=len(anchors)
        
        # numel() Returns the total number of elements in the input tensor.
        # 计算每个预测特征层上的对应的anchors数量
        num_anchors_per_level_shape_tensors=[o[0].shape for o in objectness]
        num_anchors_per_level=[s[0]*s[1]*s[2] for s in num_anchors_per_level_shape_tensors]

        objectness,pred_bbox_deltas=concat_box_prediction_layers(objectness,pred_bbox_deltas)

def concat_box_prediction_layers(box_cls,box_regression):
    # type: (List[Tensor],List[Tensor]) -> Tuple[Tensor,Tensor]

    box_cls_flattened=[]
    box_regression_flattened=[]

    for box_cls_per_level,box_regression_per_level in zip(box_cls,box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width]
        # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景      
        N,AxC,H,W=box_cls_per_level.shape
        # # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4=box_regression_per_level.shape[1]
        # anchors_num_per_position
        A=Ax4//4
        C=AxC//A

        box_cls_per_level=permute_and_flatten(box_cls_per_level,N,A,C,H,W)

def permute_and_flatten(layer,N,A,C,H,W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    """
    调整tensor顺序，并进行reshape
    Args:
        layer: 预测特征层上预测的目标概率或bboxes regression参数
        N: batch_size
        A: anchors_num_per_position
        C: classes_num or 4(bbox coordinate)
        H: height
        W: width

    Returns:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    """
    # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    # view函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    # reshape则不需要依赖目标tensor是否在内存中是连续的
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer=layer.view(N,-1,C,H,W)

    
