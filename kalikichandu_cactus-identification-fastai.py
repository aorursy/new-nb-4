# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai import *

import torch

from fastai.vision import *

train_df = pd.read_csv("../input/train.csv")

train_df.head()

test_df = pd.read_csv("../input/sample_submission.csv")

test_img = ImageList.from_df(test_df, path="../input", folder='test/test')

tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
data = (ImageList.from_df(train_df,path="../input/train",folder="train")

        .split_by_rand_pct()

#         .split_none()

        .label_from_df()

        .add_test(test_img)

        .transform(tfms,size = 128)

        .databunch(path='.',bs=64)    

        .normalize(imagenet_stats)

       )
data.show_batch(rows=3,figsize = (5,5))
data.valid_ds.classes
model = cnn_learner(data,models.resnet152, metrics = [accuracy])
model.summary()
lr = 1e-5

model.fit_one_cycle(5,lr)
# model.unfreeze()

# model.fit_one_cycle(10)
model.lr_find()

model.recorder.plot(suggestion = True)
# lr = 1e-5

model.fit_one_cycle(10,max_lr=slice(1e-2,1e-4))
model.fit_one_cycle(10,max_lr=slice(1e-2,1e-4))
model.fit_one_cycle(10,max_lr=slice(1e-2,1e-4))
model.save('stage-1-resnet152')
model.recorder.plot_losses()
interpreter = ClassificationInterpretation.from_learner(model)

interpreter.plot_confusion_matrix()
img = model.data.test_ds

img
preds, _ = model.get_preds(ds_type=DatasetType.Test)

test_df.has_cactus = preds.numpy()[:,0]

test_df.to_csv('submission.csv', index=False)