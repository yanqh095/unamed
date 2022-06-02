import json
import numpy as np
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np
from PIL import Image
from skimage.draw import polygon2mask
import pycocotools.mask as mask_util

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes

def load_label_datadict(json_file, image_root):
    """
    以json_file为准
    filename, artery, slc, bbox, lumen, outer
    file_path = image_root + filename
    supervised branch & validation
    """
    dataset_dicts = []
    with open(json_file, 'r') as f:
        data = json.load(f)
    images = data['images']
    annos = {d['image_id']:d for d in data['annotations']}
    num_of_sample = len(annos)
    for i in range(num_of_sample):
        record = {}
        image_id = images[i]['id']
        annoi = annos[image_id]
        record['file_name'] = images[i]['file_name']
        record['image_id'] = images[i]['id']
        record['height'] = images[i]['height']
        record['width'] = images[i]['width']
        
        objs = []
        obj = {}
        obj['bbox'] = annoi['bbox']
        obj['bbox_mode'] = BoxMode.XYWH_ABS
        obj['area'] = annoi['area']
        obj['category_id'] = annoi['category_id']
        obj['segmentation'] = annoi['segmentation']
        objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


def load_unlabel_datadict(json_file, image_root):
    """
    以image_root为准，实际上也并非按照slice顺序
    filename, artery, slc, annotation=None if not
    unsupervised branch & test
    """
    dataset_dicts = []
    with open(json_file, 'r') as f:
        data = json.load(f)
    annos = {d['image_id']:d for d in data['annotations']} 
    image_info = {d['id']:d for d in data['images']}
    height = {}
    width = {}
    for image_id in sorted(os.listdir(image_root)):
        if image_id.split('.')[-1] in ['jpg', 'png']:
            record = {}
            file_name = os.path.join(image_root, image_id)
            record['file_name'] = file_name
            artery = int(image_id[1:3])
            slc = int(image_id.split('_')[1].split('.')[0])
            image_id = int('1%02d%04d'%(artery, slc))
            record['image_id'] = image_id
            record['height'] = 160
            record['width'] = 320
            objs = None
            record['annotations'] = objs
            if image_id not in annos:
                dataset_dicts.append(record)
    return dataset_dicts

_datasets = {}

for i in range(5):
    _datasets['fold%d_train'%i] = ('/data1/qhong/seg/detectron2/arteryseg/artery/%02d/annotations/artery_train.json'%i,
     '/data1/qhong/seg/detectron2/arteryseg/artery/%02d/train'%i)
    _datasets['fold%d_test'%i] = ('/data1/qhong/seg/detectron2/arteryseg/artery/%02d/annotations/artery_test.json'%i,
     '/data1/qhong/seg/detectron2/arteryseg/artery/%02d/test'%i)
    

def load_artery_train(name, json_file, image_root):
    label = load_label_datadict(json_file, image_root)
    unlabel = load_unlabel_datadict(json_file, image_root)
    return (label, unlabel)

def load_artery_test(name, json_file, image_root):
    datadict = load_label_datadict(json_file, image_root)
    return datadict
    
def register_single(name, json_file, image_root, t):
    if t == 'train':
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
        DatasetCatalog.register(
                        name, lambda: load_artery_train(name, json_file, image_root)
            )
    if t == 'test':
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
        DatasetCatalog.register(
                        name, lambda: load_artery_test(name, json_file, image_root)
            )
    MetadataCatalog.get(name).thing_classes = ['artery']
            
def register_dataset():
    for key, (json_file, image_root) in _datasets.items():
        t = key.split('_')[-1]
        register_single(key, json_file, image_root, t)

register_dataset()