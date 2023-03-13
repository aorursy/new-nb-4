# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import transformers

from tokenizers import BertWordPieceTokenizer

import matplotlib.pyplot as plt
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

    """

    Tokenize text

    Source: https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []



    for i in range(0, len(texts), chunk_size):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])



    return np.array(all_ids)
def build_model(transformer, max_len=512):

    """

    Model initalization

    Source: https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    dense_layer = Dense(224, activation='relu')(cls_token)

    out = Dense(1, activation='sigmoid')(dense_layer)



    model = Model(inputs=input_word_ids, outputs=out)

    # model = InceptionV3(input_tensor=input_word_ids, weights='imagenet', include_top=True)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])



    return model
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()
AUTO = tf.data.experimental.AUTOTUNE

EPOCHS = 3

BATCH_SIZE = 32 * strategy.num_replicas_in_sync

MAX_LEN = 192
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

tokenizer.save_pretrained('.')

fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=True)
DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"





train1 = pd.read_csv(DATA_PATH+"/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv(DATA_PATH+"/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv(DATA_PATH+'/validation.csv')

test = pd.read_csv(DATA_PATH+'/test.csv')

sub = pd.read_csv(DATA_PATH+'/sample_submission.csv')

train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=150000, random_state=3982)

])
x_train = fast_encode(train.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)
y_train = train.toxic.values

y_valid = valid.toxic.values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
with strategy.scope():

    transformer_layer = (

        transformers.TFDistilBertModel

        .from_pretrained('distilbert-base-multilingual-cased')

    )

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
n_steps = x_train.shape[0] // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)
plt.plot(train_history.history['accuracy'])

plt.plot(train_history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
n_steps = x_valid.shape[0] // BATCH_SIZE

train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=EPOCHS*2

)
plt.plot(train_history_2.history['accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train'], loc='upper left')

plt.show()
sub['toxic'] = model.predict(test_dataset, verbose=1)

sub.to_csv('submission.csv', index=False)