import torch
import copy
import cv2

def IoU(box1, box2):
    """
    box: lxkx4
    """
    ys = torch.maximum(box1[:,:,:,0], box2[:,:,:,0])
    xs = torch.maximum(box1[:,:,:,1], box2[:,:,:,1])
    ye = torch.minimum(box1[:,:,:,2], box2[:,:,:,2])
    xe = torch.minimum(box1[:,:,:,3], box2[:,:,:,3])
    overlap = (ye-ys) * (xe-xs)
    overlap = torch.where(ye-ys>0, overlap, torch.tensor(0, dtype=xe.dtype, device=xe.device))
    overlap = torch.where(xe-xs>0, overlap, torch.tensor(0, dtype=xe.dtype, device=xe.device))
    area1 = (box1[:,:,:,3]-box1[:,:,:,1])*(box1[:,:,:,2]-box1[:,:,:,0])
    area2 = (box2[:,:,:,3]-box2[:,:,:,1])*(box2[:,:,:,2]-box2[:,:,:,0])
    res = overlap/(area1+area2-overlap)
    return res

def vis_box(img, boxes, color):
    grayImage = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    for box in boxes:
        cv2.rectangle(grayImage, (box[0], box[1]), (box[2],box[3]), color, 2)
    return grayImage
