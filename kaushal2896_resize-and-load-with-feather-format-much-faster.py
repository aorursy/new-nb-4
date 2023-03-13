# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import cv2

import gc

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

import time
start_time = time.time()

data0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

data1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

data2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

data3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')

print("--- %s seconds ---" % (time.time() - start_time))
def resize(df, size=128):

    resized = {}

    resize_size=128

    df = df.set_index('image_id')

    for i in tqdm(range(df.shape[0])):

        image=df.loc[df.index[i]].values.reshape(137,236)

        _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



        idx = 0 

        ls_xmin = []

        ls_ymin = []

        ls_xmax = []

        ls_ymax = []

        for cnt in contours:

            idx += 1

            x,y,w,h = cv2.boundingRect(cnt)

            ls_xmin.append(x)

            ls_ymin.append(y)

            ls_xmax.append(x + w)

            ls_ymax.append(y + h)

        xmin = min(ls_xmin)

        ymin = min(ls_ymin)

        xmax = max(ls_xmax)

        ymax = max(ls_ymax)



        roi = image[ymin:ymax,xmin:xmax]

        resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)

        resized[df.index[i]] = resized_roi.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
data0 = resize(data0)

data0.to_feather('train_data_0.feather')

del data0

data1 = resize(data1)

data1.to_feather('train_data_1.feather')

del data1

data2 = resize(data2)

data2.to_feather('train_data_2.feather')

del data2

data3 = resize(data3)

data3.to_feather('train_data_3.feather')

del data3
start_time = time.time()

data0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_0.parquet')

data1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_1.parquet')

data2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_2.parquet')

data3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_3.parquet')

print("--- %s seconds ---" % (time.time() - start_time))
data0 = resize(data0)

data0.to_feather('test_data_0.feather')

del data0

data1 = resize(data1)

data1.to_feather('test_data_1.feather')

del data1

data2 = resize(data2)

data2.to_feather('test_data_2.feather')

del data2

data3 = resize(data3)

data3.to_feather('test_data_3.feather')

del data3
start_time = time.time()

data0 = pd.read_feather('train_data_0.feather')

data1 = pd.read_feather('train_data_1.feather')

data2 = pd.read_feather('train_data_2.feather')

data3 = pd.read_feather('train_data_3.feather')

print("--- %s seconds ---" % (time.time() - start_time))
def Grapheme_plot(df):

    df_sample = df.sample(15)

    im_id, img = df_sample.iloc[:,0].values,df_sample.iloc[:,1:].values.astype(np.float)

    

    fig,ax = plt.subplots(3,5,figsize=(20,20))

    for i in range(15):

        j=i%3

        k=i//3

        ax[j,k].imshow(img[i].reshape(128,128), cmap='gray')

        ax[j,k].set_title(im_id[i],fontsize=20)

    plt.tight_layout()

        
Grapheme_plot(data0)

Grapheme_plot(data1)

Grapheme_plot(data2)

Grapheme_plot(data3)