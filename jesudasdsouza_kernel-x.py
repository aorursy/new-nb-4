import os, re

import tensorflow as tf

import pandas as pd

import numpy as np





import tensorflow as tf

import tensorflow.keras.layers as L

import tensorflow.keras.optimizers as Optim

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers

from transformers import TFAutoModel, AutoTokenizer

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
def fast_encode(texts, tokenizer, chunk_size = 256, maxlen = 512):

    tokenizer.enable_truncation(max_lenght = maxlen)

    tokenizer.enable_padding(max_length = maxlen)

    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tocanizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

        

    return np.array(all_ids)    
def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])

def build_model(transformer, max_len = 512):

    input_word_ids = L.Input(shape = (max_len,), dtype = tf.int32, name = 'input_word_ids')

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = L.Dense(1, activation = 'sigmoid')(cls_token)

    

    model = Model(inputs = input_word_ids, outputs = out)

    model.compile(Optim.Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

TRAIN_EPOCHS = 5

VALID_EPOCHS = 5

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192

MODEL = 'jplu/tf-xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)

])

x_train = regular_encode(train.comment_text.values, tokenizer, maxlen = MAX_LEN)

x_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen = MAX_LEN)

#x_test = regular_encode(test.content.values, tokenizer, maxlen = MAX_LEN)



y_train = train.toxic.values

y_valid = valid.toxic.values
train_dataset = (tf.data.Dataset

                .from_tensor_slices((x_train, y_train))

                .repeat()

                .shuffle(2048)

                .batch(BATCH_SIZE)

                .prefetch(AUTO)

                )



valid_dataset = (tf.data.Dataset

                .from_tensor_slices((x_valid, y_valid))

                .cache()

                .batch(BATCH_SIZE)

                .prefetch(AUTO)

                )



"""test_dataset = (tf.data.Dataset

                .from_tensor_slices(x_test)

                .batch(BATCH_SIZE)

                )"""

with strategy.scope():

    transformer_layer = TFAutoModel.from_pretrained(MODEL)

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
n_steps = x_train.shape[0] // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=5

)
n_steps = x_valid.shape[0] // BATCH_SIZE

train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=5

)
model.save_weights('UnToxik_V0-1.h5')
tokenizer.save_pretrained('.')