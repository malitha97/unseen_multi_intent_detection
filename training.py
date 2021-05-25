# Packages for training
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from model import *
from  pyod.models.loci import LOCI
import time
start_time = time.time()

#configs
proportion = 75 # [25, 50, 75] Different proportion of seen class
embedding_path = 'Data/glove.6B/glove.6B.300d.txt'
MAX_SEQ_LEN = None
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 300

def load_data(file_name):
    texts = []
    labels = []
    df_intial = pd.read_csv(r'Data/MixAtis/'+file_name+'.csv')
    df_intial_new = pd.DataFrame()
    for i,row in df_intial.iterrows():
      intents = row['Intents'].split(",")
      if len(intents) <= 2:
        if 'atis_flight' not in intents:
          df_intial_new = df_intial_new.append(row)
    lines = df_intial_new['Utterence']
    intents = df_intial_new['Intents']
    texts.extend(lines)
    labels.extend(intents)
    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df

def get_tokenizer():
  # loading
  with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  return tokenizer

def make_tokens(text, tokenizer):
  tokenizer_new = get_tokenizer()
  seq = tokenizer.texts_to_sequences(text)
  seq_pad = pad_sequences(seq, maxlen=MAX_SEQ_LEN, truncating='post', padding='post')
  return seq_pad

def remove_spaces(text):
    text=text.split(",")
    no_space=[]
    for sentence in text:
        sentence=sentence.lstrip()
        no_space.append(sentence)
    return no_space


def get_all(x):
    all_intents = []
    for item in x:
        all_intents.append(item)

    flat_all_intents = [item for sublist in all_intents for item in sublist]

    return pd.DataFrame(flat_all_intents)

#load data
train_df = load_data('new_train')
valid_df = load_data('valid')

#train tokenizer
import nltk
nltk.download('punkt')
train_df['content_words'] = train_df['text'].apply(lambda s: word_tokenize(s))
texts = train_df['content_words'].apply(lambda l: " ".join(l))
# Do not filter out "," and "."
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~')
tokenizer.fit_on_texts(texts)
# saving tokenizer
with open('Models/tokenizer.pickle', 'wb') as handle:
  pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


tokenizer = get_tokenizer()
word_index = tokenizer.word_index

#tokenizing utterences for training and validation
sequences_pad_train = make_tokens(train_df['text'], tokenizer)
sequences_pad_valid = make_tokens(valid_df['text'], tokenizer)
X_train = sequences_pad_train
X_valid = sequences_pad_valid

#get y labels for training and validation
y_train = train_df.label.reset_index(drop=True)
y_valid = valid_df.label.reset_index(drop=True)
print("train : valid = %d : %d" % (X_train.shape[0], X_valid.shape[0]))

print("Load pre-trained GloVe embedding...")
MAX_FEATURES = min(MAX_NUM_WORDS, len(word_index)) + 1  # +1 for PAD

def get_coefficients(wrd,*arr): return wrd, np.asarray(arr, dtype='float32')
word_embedding_index = dict(get_coefficients(*o.strip().split()) for o in open(embedding_path, encoding="utf8"))
total_embedding = np.stack(word_embedding_index.values())
emb_mean, emb_std = total_embedding.mean(), total_embedding.std()
emb_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES, EMBEDDING_DIM))
for wrd, index in word_index.items():
    if index >= MAX_FEATURES: continue
    emb_vector = word_embedding_index.get(wrd)
    if emb_vector is not None: emb_matrix[index] = emb_vector

y_train = y_train.apply(remove_spaces)
y_valid = y_valid.apply(remove_spaces)

n_class = get_all(y_train)[0].unique().shape[0]
n_class_seen = round(n_class * proportion/100)

weighted_random_sampling = False
if weighted_random_sampling:
    y_cols = get_all(y_train)[0].unique()
    y_vc = get_all(y_train)[0].value_counts()
    y_vc = y_vc / y_vc.sum()
    y_cols_seen = np.random.choice(y_vc.index, n_class_seen, p=y_vc.values, replace=False)
    y_cols_unseen = [y_col for y_col in y_cols if y_col not in y_cols_seen]
else:
    y_cols_seen = get_all(y_train)[0].value_counts().index[:n_class_seen]
    y_cols_unseen = get_all(y_train)[0].value_counts().index[n_class_seen:]


def get_seen_point_index(df):
  df_index = []
  for index, value in df.items():
    value_set = set(value)
    if value_set.issubset(set(y_cols_seen)):
      df_index.append(index)
  return np.array(df_index)

def mask_unseen(df):
  df_unseen = {}
  for index, value in df.items():
    value_set = set(value)
    intersection_value = value_set.intersection(set(y_cols_seen))
    if len(intersection_value) == 2:
      df_unseen[index] = value
    elif len(intersection_value) == 1:
      df_unseen[index] = ['unseen']+list(intersection_value)
    else:
      df_unseen[index] = ['unseen']

  return pd.Series(df_unseen)


train_seen_idx = get_seen_point_index(y_train)
valid_seen_idx = get_seen_point_index(y_valid)

X_train_seen = X_train[train_seen_idx]
y_train_seen = y_train[train_seen_idx]
X_valid_seen = X_valid[valid_seen_idx]
y_valid_seen = y_valid[valid_seen_idx]

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(y_train_seen)

# saving ohe
with open('Models/ohe.pickle', 'wb') as handle:
  pickle.dump(multilabel_binarizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading ohe
with open('Models/ohe.pickle', 'rb') as handle:
  ohe = pickle.load(handle)

y_train_onehot = ohe.transform(y_train_seen)
y_valid_onehot = ohe.transform(y_valid_seen)

y_train_onehot = tf.cast(y_train_onehot, tf.float32)
y_valid_onehot = tf.cast(y_valid_onehot, tf.float32)
X_train_seen = tf.cast(X_train_seen, tf.float32)
X_valid_seen = tf.cast(X_valid_seen, tf.float32)

filepath = 'Data/BiLSTM_' + 'MixAtis' + "_" + str(proportion) + '-AM.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                             save_best_only=True, mode='auto', save_weights_only=False)
early_stop = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
callbacks_list = [checkpoint, early_stop]

## building the model
model = build_model(MAX_SEQ_LEN, MAX_FEATURES, EMBEDDING_DIM, n_class_seen, None, emb_matrix)

history = model.fit(X_train_seen, y_train_onehot, epochs=10, batch_size=128,
                    validation_data=(X_valid_seen, y_valid_onehot), shuffle=True, verbose=2, callbacks=callbacks_list)

#Saving the model
model.save('Models/main_model.h5')

#for the training
get_deep_feature = Model(inputs=model.input,
                         outputs=model.layers[-3].output)
feature_train = get_deep_feature.predict(X_train_seen)

method = 'LOF (LMCL)'
lof = LocalOutlierFactor(n_neighbors=17, contamination=0.05, novelty=True, n_jobs=-1)
lof.fit(feature_train)

#Saving the model to get deep features
get_deep_feature.save('Models/get_deep_feature.h5')

#Saving the model LOF
with open('Models/lof.pickle', 'wb') as handle:
  pickle.dump(lof, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("--- %s seconds ---" % (time.time() - start_time))