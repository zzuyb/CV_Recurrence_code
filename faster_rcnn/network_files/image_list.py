from ctypes import cast
from typing import List,Tuple

from torch import Tensor, device

class ImageList(object):
    """
    
    """
    def __init__(self,tensors,image_sizes):
        # type: (Tensor,List[Tuple[int,int]]) -> None

        """
        Arguments"
            tensors (tensor): after padding 
            image_sizes (List[Tuple[int,int]]) :before padding
        """

        self.tensors=tensors
        self.image_sizes=image_sizes
    
    def to(self,devcie):
        # type: (device) -> ImageList # noqa

        cast_tensor=self.tensors.to(devcie)
        return ImageList(cast_tensor,self.image_sizes)
        