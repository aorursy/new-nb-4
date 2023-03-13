import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from skimage.io import imread # read image

from PIL import Image 

# imread fails on some of the tiffs so we use PIL

pil_imread = lambda c_file: np.array(Image.open(c_file)) 

from skimage.exposure import equalize_adapthist

from glob import glob




import matplotlib.pyplot as plt
list_train = glob(os.path.join('..', 'input', 'train', '*', '*.jpg'))

print('Train Files found', len(list_train), 'first file:', list_train[0])

list_test = glob(os.path.join('..', 'input', '*', '*.tif'))

print('Test Files found', len(list_test), 'first file:', list_test[0])
from sklearn.preprocessing import LabelEncoder

def get_class_from_path(filepath):

    return os.path.dirname(filepath).split(os.sep)[-1]

full_train_df = pd.DataFrame([{'path': x, 'category': get_class_from_path(x)} for x in list_train])

cat_encoder = LabelEncoder()

cat_encoder.fit(full_train_df['category'])

nclass = cat_encoder.classes_.shape[0]

full_train_df.sample(3)
fig, ax1 = plt.subplots(1,1,figsize = (8, 6))

ax1.hist(cat_encoder.transform(full_train_df['category']), np.arange(nclass+1))

ax1.set_xticks(np.arange(nclass))

_ = ax1.set_xticklabels(cat_encoder.classes_, rotation = 45)
import cv2

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))

def imread_and_normalize(im_path):

    img_data = pil_imread(im_path)

    img_data = cv2.cvtColor(img_data[:,:,[2,1,0]], cv2.COLOR_BGR2LAB)

    img_data[:,:,0] = clahe.apply(img_data[:,:,0])

    img_data = cv2.cvtColor(img_data, cv2.COLOR_LAB2BGR)

    # don't run channel by channel

    #for i in range(3):

    #    img_data[:,:,i] = clahe.apply(img_data[:,:,i])

    return (img_data.astype(np.float32))/255.0

# code for reading in a random chunk of the image

def read_chunk(im_path, n_chunk = 5, chunk_x = 96, chunk_y = 96):

    img_data = imread_and_normalize(im_path)

    img_x, img_y, _ = img_data.shape

    out_chunk = []

    for _ in range(n_chunk):

        x_pos = np.random.choice(range(img_x-chunk_x))

        y_pos = np.random.choice(range(img_y-chunk_y))

        out_chunk += [img_data[x_pos:(x_pos+chunk_x), y_pos:(y_pos+chunk_y),:3]]

    return np.stack(out_chunk, 0)



t_img = read_chunk(full_train_df['path'].values[0])

fig, c_axs = plt.subplots(2, 3, figsize = (12, 4))

for i, (c_ax, m_ax) in enumerate(c_axs.T):

    c_ax.imshow(t_img[0,:,:,i], interpolation='none')

    c_ax.axis('off')

    c_ax.set_title(['Red', 'Green', 'Blue'][i])

    m_ax.hist(t_img[0,:,:,i].ravel())
from keras.utils.np_utils import to_categorical

def generate_even_batch(base_df, sample_count = 1, chunk_count = 5):

    while True:

        cur_df = base_df.groupby('category').apply(lambda x: x[['path']].sample(sample_count)).reset_index()

        x_out = np.concatenate(cur_df['path'].map(lambda x: read_chunk(x, n_chunk=chunk_count)),

                             0)

        y_raw = [x for x in cur_df['category'].values for _ in range(chunk_count)]

        y_out = to_categorical(cat_encoder.transform(y_raw))

        yield x_out, y_out
d_gen = generate_even_batch(full_train_df)

for _, (x, y) in zip(range(1), d_gen):

    print(x.shape, y.shape)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from keras import optimizers, losses, activations, models

from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate

def gap_drop(in_layer): 

    gap_layer = GlobalAveragePooling2D()(Convolution2D(16, kernel_size = 1)(in_layer))

    gmp_layer = GlobalMaxPool2D()(Convolution2D(16, kernel_size = 1)(in_layer))

    return Dropout(rate = 0.5)(concatenate([gap_layer, gmp_layer]))



def create_model():

    inp = Input(shape=(None, None, 3))

    norm_inp = BatchNormalization()(inp)

    gap_layers = []

    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(norm_inp)

    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(img_1)

    #gap_layers += [gap_drop(img_1)]

    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)

    img_1 = Dropout(rate=0.2)(img_1)

    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)

    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)

    gap_layers += [gap_drop(img_1)]

    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)

    img_1 = Dropout(rate=0.2)(img_1)

    img_1 = Convolution2D(64, kernel_size=2, activation=activations.relu, padding="same")(img_1)

    img_1 = Convolution2D(64, kernel_size=2, activation=activations.relu, padding="same")(img_1)

    gap_layers += [gap_drop(img_1)]

    

    gap_cat = concatenate(gap_layers)

    

    dense_1 = Dense(32, activation=activations.relu)(gap_cat)

    dense_1 = Dense(nclass, activation='softmax')(dense_1)



    model = models.Model(inputs=inp, outputs=dense_1)

    opt = optimizers.Adam(lr=1e-3) # karpathy's magic learning rate

    model.compile(optimizer=opt, 

                  loss='categorical_crossentropy', 

                  metrics=['acc'])

    model.summary()

    return model

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(full_train_df, 

                                     test_size = 0.15,

                                    random_state = 2018,

                                    stratify = full_train_df['category'])

print('Train', train_df.shape[0], 'Test', test_df.shape[0])

train_gen = generate_even_batch(train_df, 3, chunk_count = 3)

test_gen = generate_even_batch(test_df, 10)

# cache the test_gen_data

(test_x, test_y) = next(test_gen)

print('Test Data', test_x.shape)
model = create_model()

file_path="weights.best.hdf5"



checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')



early = EarlyStopping(monitor="val_acc", mode="max", patience=3)

callbacks_list = [checkpoint, early] #early
history = model.fit_generator(train_gen, 

                              steps_per_epoch = 10,

                              validation_data = (test_x, test_y), 

                              epochs = 10, 

                              verbose = True,

                              workers = 2,

                              use_multiprocessing = False,

                              callbacks = callbacks_list)



#print(history)



model.load_weights(file_path)
# show the processed image

t_img = imread_and_normalize(np.random.choice(list_test))

fig, c_axs = plt.subplots(2, 3, figsize = (12, 4))

for i, (c_ax, m_ax) in enumerate(c_axs.T):

    c_ax.imshow(t_img[:,:,i], interpolation='none')

    c_ax.axis('off')

    m_ax.hist(t_img[:,:,i].ravel())
from tqdm import tqdm

out_dict_list = []

for c_file in tqdm(list_test):

    img_data = imread_and_normalize(c_file)

    n_image = np.expand_dims(img_data,0)

    out_dict_list += [{

        'fname': os.path.basename(c_file),

        'camera': np.argmax(model.predict(n_image)[0])

    }]  
df = pd.DataFrame(out_dict_list)

df['camera'] = df['camera'].map(cat_encoder.inverse_transform)

df[['fname', 'camera']].to_csv("submission.csv", index=False)

df.sample(3)
fig, ax1 = plt.subplots(1,1,figsize = (8, 6))

ax1.hist(cat_encoder.transform(df['camera']), np.arange(nclass+1))

ax1.set_xticks(np.arange(nclass)+0.5)

_ = ax1.set_xticklabels(cat_encoder.classes_, rotation = 90)