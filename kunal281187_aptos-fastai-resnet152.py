

import fastai

from fastai import *

from fastai.vision import *

from fastai.callbacks import *

import cv2



import pandas as pd

import matplotlib.pyplot as plt
# Making pretrained weights work without needing to find the default filename

if not os.path.exists('/tmp/.cache/torch/checkpoints/'):

        os.makedirs('/tmp/.cache/torch/checkpoints/')



#!cp '../input/densenet161/densenet161-8d451a50.pth' '/tmp/.cache/torch/checkpoints/densenet161-8d451a50.pth'
import os

os.listdir('../input')
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 1667

seed_everything(SEED)
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')

train_dir = os.path.join(base_image_dir,'train_images/')

df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))

df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

df = df.drop(columns=['id_code'])

df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

df.head(10)
len_df = len(df)

print(f"There are {len_df} images")
df['diagnosis'].hist(figsize = (10, 5))
def crop_image1(img,tol=7):

    # img is image data

    # tol  is tolerance

        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]



def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img



def load_ben_color(path, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        

    return image
   

src = (ImageList.from_df(df=df,path='./',cols='path') #get dataset from dataset

        .split_by_rand_pct(0.2, seed=42) #Splitting the dataset

        .label_from_df(cols='diagnosis',label_cls=FloatList) #obtain labels from the level column

      )

src
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0.10, max_zoom=1.3, max_warp=0.0, max_lighting=0.2)

data = (

    src.transform(tfms,size=224)

    .databunch()

    .normalize(imagenet_stats)

)

data
from sklearn.metrics import cohen_kappa_score

def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'),device='cuda:0')
#update - kappa from fastai

kappa = KappaScore()

kappa.weights = "quadratic"

#learn = cnn_learner(data, arch, metrics=kappa)
learn = cnn_learner(data, base_arch=models.resnet152 ,

                    metrics=[quadratic_kappa],model_dir='/kaggle',

                    pretrained=True,

                    callback_fns=[partial(EarlyStoppingCallback, monitor='quadratic_kappa', 

                                          min_delta=0.01, patience=3)])
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(6, 3e-2)

learn.save('stage1')
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-3))
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.export()

learn.save('stage2')

learn.show_results(ds_type=DatasetType.Train, rows=4, figsize=(8,10))
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



#len(data.valid_ds)==len(losses)==len(idxs)

#interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
valid_preds = learn.get_preds(ds_type=DatasetType.Valid)
import numpy as np

import pandas as pd

import os

import scipy as sp

from functools import partial

from sklearn import metrics

from collections import Counter

import json
class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

        print(-loss_partial(self.coef_['x']))



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']
optR = OptimizedRounder()

optR.fit(valid_preds[0],valid_preds[1])
coefficients = optR.coefficients()

print(coefficients)
from fastai.core import *

from fastai.basic_data import *

from fastai.basic_train import *

from fastai.torch_core import *

def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid, num_pred:int=10) -> Iterator[List[Tensor]]:

    "Computes the outputs for several augmented inputs for TTA"

    dl = learn.dl(ds_type)

    ds = dl.dataset

    old = ds.tfms

    aug_tfms = [o for o in learn.data.train_ds.tfms]

    try:

        pbar = master_bar(range(num_pred))

        for i in pbar:

            ds.tfms = aug_tfms

            yield get_preds(learn.model, dl, pbar=pbar)[0]

    finally: ds.tfms = old



Learner.tta_only = _tta_only



def _TTA(learn:Learner, beta:float=0, ds_type:DatasetType=DatasetType.Valid, num_pred:int=10, with_loss:bool=False) -> Tensors:

    "Applies TTA to predict on `ds_type` dataset."

    preds,y = learn.get_preds(ds_type)

    all_preds = list(learn.tta_only(ds_type=ds_type, num_pred=num_pred))

    avg_preds = torch.stack(all_preds).mean(0)

    if beta is None: return preds,avg_preds,y

    else:            

        final_preds = preds*beta + avg_preds*(1-beta)

        if with_loss: 

            with NoneReduceOnCPU(learn.loss_func) as lf: loss = lf(final_preds, y)

            return final_preds, y, loss

        return final_preds, y



Learner.TTA = _TTA
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection',folder='test_images',suffix='.png'))

preds,y = learn.get_preds(ds_type=DatasetType.Test)

test_predictions = optR.predict(preds, coefficients)

sample_df.diagnosis = test_predictions.astype(int)

sample_df.head()

sample_df.to_csv('submission.csv',index=False)