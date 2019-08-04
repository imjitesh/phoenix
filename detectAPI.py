from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import json

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

def detection(folder):
    opt = {
        "image_folder":folder,
        "model_def":"config/yolov3.cfg",
        "weights_path":"weights/yolov3.weights",
        "class_path":"data/coco.names",
        "conf_thres":0.8,
        "nms_thres":0.4,
        "batch_size":1,
        "n_cpu":0,
        "img_size":416
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt['model_def'], img_size=opt['img_size']).to(device)

    if opt['weights_path'].endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt['weights_path'])
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt['weights_path']))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt['image_folder'], img_size=opt['img_size']),
        batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=opt['n_cpu'],
    )

    classes = load_classes(opt['class_path'])  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt['conf_thres'], opt['nms_thres'])
        imgs.extend(img_paths)
        img_detections.extend(detections)

    send_detections = []

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        source_img = Image.open(path).convert("RGBA")
        draw = ImageDraw.Draw(source_img)
        obj_detections = {'image_url':path,'annotations':[],'image_width':source_img.size[0],'image_height':source_img.size[1]}
        if detections is not None:
            detections = rescale_boxes(detections, opt['img_size'], (source_img.size[1],source_img.size[0]))
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                #print("\t- %.5f %.5f %.5f %.5f" % (x1,y1,x2,y2))
                #draw.rectangle(((x1, y1), (x2, y2)), fill=(255,0,0,127))
                draw.line((x1,y1, x2, y1), fill="#00dd00", width=1)
                draw.line((x2,y1, x2, y2), fill="#00dd00", width=1)
                draw.line((x2,y2, x1, y2), fill="#00dd00", width=1)
                draw.line((x1,y2, x1, y1), fill="#00dd00", width=1)
                obj_detections['annotations'].append({'class':classes[int(cls_pred)],'coordinates':[{'x':x1.item(),'y':y1.item()},{'x':x2.item(),'y':y2.item()}]})
            send_detections.append(obj_detections)

        filename = path.split("/")[-1].split(".")[0]
        source_img.save('output/'+filename+".png", "png")
    return send_detections