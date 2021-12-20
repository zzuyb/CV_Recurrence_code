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
        self.root=os.path.join(voc_root,"VOCdevkit",f"VOC{year}")#不明白
        self.img_root=os.path.join(self.root,"JPEGImages")
        self.annotation_root=os.path.join(self.root,"Annotation")

        txt_path=os.path.join(self.root,"ImageSets","Main",txt_name)
        assert os.path.exists(txt_path),"not found{} file.".format(txt_name)

        with open(txt_path) as read:
            self.xml_list=[os.path.join(self.annotation_root),line.strip()+".xml" 
                           for line in read.readlines() if len(line.strip())>0]
        
        assert len(self.xml_list)>0,"in '{}' file does not find any information".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path),"not found '{}' file.".format(xml_path)

        # read class_indict
        json_file='./pascal_voc_classes.json'
        assert os.path.exists(json_file),"{} file not exist".format(json_file)
        json_file=open(json_file,'r')
        self.class_dict=json.load(json)
        json_file.close()

        self.transforms=transforms

    def __len__(self) -> int:
        return len(self.xml_list)

    def __getitem__(self, idx: int):
        xml_path=self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str=fid.read()#read xml content
        
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)["annotation"]
        img_path=os.path.join(self.img_root,data["filename"])
        image=Image.open(img_path)
        if image.format != 'JPEG':
            raise ValueError("Image '{}' format not JPEG".format(img_path))
        
        boxes=[]
        labels=[]
        iscrowd=[]# judge difficult
        assert "object" in data,"{} lack of object information.".format(xml_path)

        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

        # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan 
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            
            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
               iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)
        
        # convert everything into a torch.Tensor
        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id=torch.tensor([idx])
        area=(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        
        target={}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image,target=self.transforms(image,target)
        
        return image,target

    def get_height_and_width(self,idx):
        # read xml
        xml_path=self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str=fid.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)["annotation"]
        data_height=int(data["size"]["height"])
        data_width=int(data["size"]["width"])
        return data_height,data_width

    def parse_xml_to_dict(self,xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.        
        """
        if len(xml)==0:
            return {xml.tag:xml.txt}

        result={}
        for child in xml:
            child_result=self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag]=child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag]=[]
                result[child.tag].append(child_result[child.tag])
        return {xml.tag:result}