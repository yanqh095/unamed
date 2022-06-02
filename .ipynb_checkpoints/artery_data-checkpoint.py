from detectron2.data.datasets import register_coco_instances
register_coco_instances("artery", {}, "/data1/qhong/seg/mmdetection/data/anno_set/00/annotations/artery_train.json", 
                        "/data1/qhong/seg/mmdetection/data/anno_set/00/train")