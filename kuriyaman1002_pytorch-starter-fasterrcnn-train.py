import pandas as pd

import numpy as np

import cv2

import os

import re

import random



from PIL import Image



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator



from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler



from matplotlib import pyplot as plt



DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_save_nm = 'fasterrcnn_resnet50_fpn_20200731.pth'

num_epochs = 100

PATIENCE = 3
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

train_df.shape
train_df['x'] = -1

train_df['y'] = -1

train_df['w'] = -1

train_df['h'] = -1



def expand_bbox(x):

    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r



train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

train_df.drop(columns=['bbox'], inplace=True)

train_df['x'] = train_df['x'].astype(np.float)

train_df['y'] = train_df['y'].astype(np.float)

train_df['w'] = train_df['w'].astype(np.float)

train_df['h'] = train_df['h'].astype(np.float)
image_ids = train_df['image_id'].unique()

random.seed(0)

random.shuffle(image_ids)

valid_ids = image_ids[-665:]

train_ids = image_ids[:-665]
valid_df = train_df[train_df['image_id'].isin(valid_ids)]

train_df = train_df[train_df['image_id'].isin(train_ids)]
valid_df.shape, train_df.shape
class WheatDataset(Dataset):



    def __init__(self, dataframe, image_dir, transforms=None):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]



        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        boxes = records[['x', 'y', 'w', 'h']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        area = torch.as_tensor(area, dtype=torch.float32)



        # there is only one class

        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        

        # suppose all instances are not crowd

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        # target['masks'] = None

        target['image_id'] = torch.tensor([index])

        target['area'] = area

        target['iscrowd'] = iscrowd



        if self.transforms:

            sample = {

                'image': image,

                'bboxes': target['boxes'],

                'labels': labels

            }

            sample = self.transforms(**sample)

            image = sample['image']

            

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)



        return image, target, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
# Albumentations

# 参考: https://www.kaggle.com/aleksandradeis/globalwheatdetection-eda/notebook

def get_train_transform():

    return A.Compose([

        A.RandomSizedBBoxSafeCrop(512, 512, erosion_rate=0.0, interpolation=1, p=1.0),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.OneOf([A.RandomContrast(),

                    A.RandomGamma(),

                    A.RandomBrightness()], p=1.0),

#         A.CLAHE(p=1.0),

        ToTensorV2(p=1.0),

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



def get_valid_transform():

    return A.Compose([

        ToTensorV2(p=1.0)

    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# load a model; pre-trained on COCO

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

# model.to(device)

# Load the trained weights

WEIGHTS_FILE = '../input/wheat-submission/fasterrcnn_resnet50_fpn.pth'

model.load_state_dict(torch.load(WEIGHTS_FILE))
num_classes = 2  # 1 class (wheat) + background



# get number of input features for the classifier

in_features = model.roi_heads.box_predictor.cls_score.in_features



# replace the pre-trained head with a new one

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
class Averager:

    def __init__(self):

        self.current_total = 0.0

        self.iterations = 0.0



    def send(self, value):

        self.current_total += value

        self.iterations += 1



    @property

    def value(self):

        if self.iterations == 0:

            return 0

        else:

            return 1.0 * self.current_total / self.iterations



    def reset(self):

        self.current_total = 0.0

        self.iterations = 0.0

def collate_fn(batch):

    return tuple(zip(*batch))



train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())

valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())





# split the dataset in train and test set

indices = torch.randperm(len(train_dataset)).tolist()



train_data_loader = DataLoader(

    train_dataset,

    batch_size=16,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)



valid_data_loader = DataLoader(

    valid_dataset,

    batch_size=8,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)
images, targets, image_ids = next(iter(train_data_loader))

images = list(image.to(device) for image in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)

sample = images[2].permute(1,2,0).cpu().numpy()
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    

ax.set_axis_off()

ax.imshow(sample)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

lr_scheduler = None

loss_hist = Averager()

loss_hist_val = Averager()

loss_list = []

loss_list_val = []

itr = 1

best_epoch = 0

min_loss_val = 1.



for epoch in range(num_epochs):

    model.train()

    loss_hist.reset()

    loss_hist_val.reset()

    

    for images, targets, image_ids in train_data_loader:

        

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        loss_dict = model(images, targets)



        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()



        loss_hist.send(loss_value)



        optimizer.zero_grad()

        losses.backward()

        optimizer.step()



        if itr % 50 == 0:

            print(f"Iteration #{itr} loss: {loss_value}")



        itr += 1

    

    # update the learning rate

    if lr_scheduler is not None:

        lr_scheduler.step()



    print(f"Epoch #{epoch} loss: {loss_hist.value}")

    

    if epoch % 5 == 0:

    #     model.eval()

        for images, targets, image_ids in valid_data_loader:



            images = list(image.to(device) for image in images)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



            loss_dict = model(images, targets)



            losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()



            loss_hist_val.send(loss_value)

        optimizer.zero_grad()

        print(f"Epoch #{epoch} loss_val: {loss_hist_val.value}")

        

        loss_list.append(loss_hist.value)

        loss_list_val.append(loss_hist_val.value)

        

        if loss_hist_val.value < min_loss_val:

            min_loss_val = loss_hist_val.value

            best_epoch = epoch

        if (epoch - best_epoch) >= PATIENCE:

            print(f'train stops at {best_epoch}th epoch')
plt.plot(loss_list)

plt.plot(loss_list_val)
images, targets, image_ids = next(iter(valid_data_loader))
images = list(img.to(device) for img in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)

sample = images[1].permute(1,2,0).cpu().numpy()
model.eval()

cpu_device = torch.device("cpu")



outputs = model(images)

outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    

ax.set_axis_off()

ax.imshow(sample)
torch.save(model.state_dict(), model_save_nm)
DIR_INPUT = '/kaggle/input/global-wheat-detection'

# DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'



test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')

test_df.shape
class WheatTestDataset(Dataset):



    def __init__(self, dataframe, image_dir, transforms=None):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]



        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        if self.transforms:

            sample = {

                'image': image,

            }

            sample = self.transforms(**sample)

            image = sample['image']



        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
# Albumentations

def get_test_transform():

    return A.Compose([

        # A.Resize(512, 512),

        ToTensorV2(p=1.0)

    ])

test_dataset = WheatTestDataset(test_df, DIR_TEST, get_test_transform())



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
detection_threshold = 0.5

results = []



for images, image_ids in test_data_loader:



    images = list(image.to(device) for image in images)

    outputs = model(images)



    for i, image in enumerate(images):



        boxes = outputs[i]['boxes'].data.cpu().numpy()

        scores = outputs[i]['scores'].data.cpu().numpy()

        

        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        scores = scores[scores >= detection_threshold]

        image_id = image_ids[i]

        

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        

        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }



        

        results.append(result)
results[0:2]
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()
sample = images[1].permute(1,2,0).cpu().numpy()

boxes = outputs[1]['boxes'].data.cpu().numpy()

scores = outputs[1]['scores'].data.cpu().numpy()



boxes = boxes[scores >= detection_threshold].astype(np.int32)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 2)

    

ax.set_axis_off()

ax.imshow(sample)
test_df.to_csv('submission.csv', index=False)

print(test_df.shape)