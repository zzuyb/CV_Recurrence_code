import math
from typing import List,Tuple,Dict,Optional
from matplotlib.pyplot import sca

import torch
from torch import nn,Tensor
import torchvision
from .image_list import ImageList

@torch.jit.unused
def _resize_image_onnx(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    from torch.onnx import operators
    im_shape = operators.shape_as_tensor(image)[-2:]
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)

    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image

def _resize_image(image,self_min_size,self_max_size):
    # type: (Tensor,float,float) -> Tensor
    im_shape=torch.tensor(image.shape[-2:])
    min_size=float(torch.min(im_shape))
    max_size=float(torch.max(im_shape))
    scale_factor=self_min_size/min_size

    if max_size*scale_factor>self_max_size:
        scale_factor=self_max_size/max_size

    image=torch.nn.functional.interpolate(
        image[None],scale_factor=scale_factor,mode="bilinear",recompute_scale_factor=True,
        align_corners=False)[0]

    return image

def resize_bboxes(boxes,original_size,new_size):
    ratios=[
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s,s_orig in zip(new_size,original_size)
    ]
    ratios_h,ratios_w=ratios

    xmin,ymin,xmax,ymax=boxes.unbind(1)
    xmin=xmin*ratios_w
    xmax=xmax*ratios_w
    ymin=ymin*ratios_h
    ymax=ymax*ratios_h
    return torch.stack((xmin,ymin,xmax,ymax),dim=1)

class GeneralizedRCNNTransform(nn.Module):
    """
    The transformations it perform are:
        - input normalization(mean and std)
        - input/target resizing to match min_size/max_size
    
    It returns a ImageList for the inputs,and a List[Dict[Tensor]] for the targets
    """
    def __init__(self,min_size,max_size,image_mean,image_std):
        super(GeneralizedRCNNTransform,self).__init__()
        if not isinstance(min_size,(list,tuple)):
            min_size=(min_size,)
        self.min_size=min_size 
        self.max_size=max_size
        self.image_mean=image_mean
        self.image_std=image_std


    def normalize(self,image):

        dtype,device=image.dtype,image.device
        mean=torch.as_tensor(self.image_mean,dtype=dtype,device=device)
        std=torch.as_tensor(self.image_std,dtype=dtype,device=device)
        # [:,None,None] shape[3]-> [3,1,1]
        return (image-mean[:,None,None])/std[:,None,None]

    def torch_choice(self,k):
        # type: (List[int]) -> int

        index=int(torch.empty(1).uniform_(0,float(len(k))).item())
        return k[index]
    
    def resize(self,image,target):
        # type: (Tensor,Optional[Dict[str,Tensor]]) -> Tuple[Tensor,Optional[Dict[str,Tensor]]]

        h,w=image.shape[-2:]

        if self.training:
            size=float(self.torch_choice(self.min_size))
        else:
            size=float(self.min_size[-1])
        
        if torchvision._is_tracing():
            image = _resize_image_onnx(image, size, float(self.max_size))
        else:
            image=_resize_image(image,size,float(self.max_size))

        if target is None:
            return image,target
        
        bbox=target["bboxes"]

        bbox=resize_bboxes(bbox,[h,w],image.shape[-2:])
        target["bboxes"]=bbox
        return image,target
    
    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, [0, padding[2], 0, padding[1], 0, padding[0]])
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self,the_list):
        # type:(List[List[int]]) -> List[int]
        maxes=the_list[0]
        for sublist in the_list[1:]:
            for index,item in enumerate(sublist):
                maxes[index]=max(item,maxes[index])
        return maxes

    def batch_images(self,images,size_divisible=32):
        # type: (List[Tensor],int) -> Tensor
        """
        
        """
        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        max_size=self.max_by_axis([list(img.shape) for img in images])
        stride=float(size_divisible)

        max_size[1]=int(math.ceil(float(max_size[1])/stride)*stride)
        max_size[2]=int(math.ceil(float(max_size[2])/stride)*stride)  

        #[batch,channel,height,width]
        batch_shape=[len(images)]+max_size

        batched_imgs=images[0].new_full(batch_shape,0)
        for img,pad_img in zip(images,batched_imgs):
            pad_img[:img.shape[0],:img.shape[1],:img.shape[2]].copy_(img)
        
        return batched_imgs
    def postprocess(self,
                    result,                 # type: List[Dict[str,Tensor]]
                    image_shapes,           # type: List[Tuple[int,int]]
                    original_image_sizes    # type: List[Tuple[int,int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """                    
        if self.training:
            return result
        
        for i,(pred,im_s,o_im_s) in enumerate(zip(result,image_shapes,original_image_sizes)):
            boxes=pred["boxes"]
            boxes=resize_bboxes(boxes,im_s,o_im_s)
            result[i]["boxes"]=boxes
        return result

    def __repr__(self):
        """自定义输出实例化对象的信息，可通过print打印实例信息"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string
        
    def forward(self,
                images,          # type: List[Tensor]
                targets=None    # type: Optional[List[Dict[str,Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList,Optional[List[Dict[str,Tensor]]]]
        images=[img for img in images]
        for i in range(len(images)):
            image=images[i]
            target_index=targets[i] if targets is not None else None
            
            if image.dim()!=3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))

            image=self.normalize(image)
            image,target_index=self.resize(image,target_index)
            images[i]=image
            if targets is not None and target_index is not None:
                targets[i]=target_index 
            
        
        image_sizes=[img.shape[-2:] for img in images]
        images=self.batch_images(images)

