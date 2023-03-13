import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout, concatenate
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
# define network parameters
max_features = 20000
maxlen = 100
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("Invalid").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("Invalid").values
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
# train data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
# test data
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
def build_model(conv_layers = 2, max_dilation_rate = 3):
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Dropout(0.25)(x)
    x = Conv1D(2*embed_size, 
                   kernel_size = 3)(x)
    prefilt_x = Conv1D(2*embed_size, 
                   kernel_size = 3)(x)
    out_conv = []
    # dilation rate lets us use ngrams and skip grams to process 
    for dilation_rate in range(max_dilation_rate):
        x = prefilt_x
        for i in range(3):
            x = Conv1D(32*2**(i), 
                       kernel_size = 3, 
                       dilation_rate = dilation_rate+1)(x)    
        out_conv += [Dropout(0.5)(GlobalMaxPool1D()(x))]
    x = concatenate(out_conv, axis = -1)    
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    return model

model = build_model()
model.summary()
batch_size = 512
epochs = 15

file_path="weights.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early] #early
model.fit(X_t, y, 
          batch_size=batch_size, 
          epochs=epochs, 
          validation_split=0.1, 
          callbacks=callbacks_list)

from IPython.display import Markdown, display
dmd = lambda x: display(Markdown(x))
def show_sentence(sent_idx):
    dmd('# Input Sentence:\n `{}`'.format(list_sentences_train[sent_idx]))
    c_pred = model.predict(X_t[sent_idx:sent_idx+1])[0]
    dmd('## Positive Categories')
    for k, v, p in zip(list_classes, y[sent_idx], c_pred):
        if v>0:
            dmd('- {}, Prediction: {:2.2f}%'.format(k, 100*v, 100*p))
    dmd('## Negative Categories')
    for k, v, p in zip(list_classes, y[sent_idx], c_pred):
        if v<1:
            dmd('- {}, Prediction: {:2.2f}%'.format(k, 100*p))
show_sentence(0)
show_sentence(50)
model.load_weights(file_path)
y_test = model.predict(X_te, verbose = True, batch_size = 1024)
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("predictions.csv", 
                         index=False)
