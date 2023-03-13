# install orderedset
import sys
sys.path.append('mmcv60') # To find local version

sys.path.append('mmdetection') # To find local version
# add to sys python path for pycocotools
sys.path.append('/opt/conda/lib/python3.7/site-packages/pycocotools-2.0-py3.7-linux-x86_64.egg') # To find local version
import mmcv

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset

import pandas as pd
import os
import json

from PIL import Image
import matplotlib.pyplot as plt
import torch
config = './mmdetection/configs/DetectoRS/WheatDetectoRS_mstrain_400_1200_r50_40e.py'
checkpoint = '/kaggle/input/detectors2/epoch_40.pth'
model = init_detector(config, checkpoint, device='cuda:0')
# model = init_detector(config, checkpoint, device='cpu')
import cv2
img = '/kaggle/input/global-wheat-detection/test/2fd875eaa.jpg'
result = inference_detector(model, img)
# img = cv2.imread(img)
print(type(img))
show_result_pyplot(img, result,['wheat'], score_thr=0.3)
result[0][0].shape

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
def gen_test_annotation(test_data_path, annotation_path):
    test_anno_dict = {}
    test_anno_dict["info"] = "jianqiu created"
    test_anno_dict["license"] = ["license"]
    id = 0
    test_anno_list = []
    for img in os.listdir(test_data_path):
        if img.endswith('jpg'):
            id += 1
            img_info = {}
            img_size = Image.open(os.path.join(test_data_path, img)).size
            img_info['height'] = img_size[1]
            img_info['width'] = img_size[0]
            img_info['id'] = id            
            img_info['file_name'] = img
            test_anno_list.append(img_info)
    test_anno_dict["images"] = test_anno_list
    test_anno_dict["categories"] = [
    {
      "id": 1,
      "name": "wheat"
    }
  ]
    with open(annotation_path, 'w+') as f:
        json.dump(test_anno_dict, f)
DIR_INPUT = '/kaggle/working/mmdetection/data/Wheatdetection'
DIR_TEST = f'{DIR_INPUT}/test'
DIR_ANNO = f'{DIR_INPUT}/annotations'

DIR_WEIGHTS = '/kaggle/input/detectors2'
WEIGHTS_FILE = f'{DIR_WEIGHTS}/epoch_40.pth'

# prepare test data annotations
gen_test_annotation(DIR_TEST, DIR_ANNO + '/detection_test.json')
config_file = '/kaggle/input/detestorstest/WheatDetectoRS_mstrain_400_1200_r50_40e.py'
cfg = Config.fromfile(config_file)
cfg.data.test.test_mode = True

distributed = False
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    imgs_per_gpu = 1,
    workers_per_gpu=1,
    dist=distributed,
    shuffle=False)
# Wbf

import sys
sys.path.insert(0, "../input/weightedboxesfusion")
from ensemble_boxes import *
import numpy as np
def run_wbf(prediction, image_size=1024, iou_thr=0.43, skip_box_thr=0.43, weights=None):
    boxes = [(prediction[:, :4]/(image_size-1)).tolist()]
    scores = [(prediction[:,4]).tolist()]
    labels = [(np.ones(prediction[:,4].shape[0])).tolist() ]

    boxes, scores, labels = nms(boxes, scores, labels, weights=None, iou_thr=iou_thr)
    boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, WEIGHTS_FILE, map_location='cpu') # 'cuda:0'

model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=[0])
outputs = single_gpu_test(model, data_loader, False)

results = []

for images_info, result in zip(dataset.img_infos, outputs):
    boxes, scores, labels = run_wbf(result[0][0])
#     boxes = result[0][0][:, :4]
#     scores = result[0][0][:, 4]
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    result = {
        'image_id': images_info['filename'][:-4],
        'PredictionString': format_prediction_string(boxes, scores)
    }

    results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

# save result
test_df.to_csv('submission.csv', index=False)
test_df.head()
len(results[4]['PredictionString'].split(' '))//5
