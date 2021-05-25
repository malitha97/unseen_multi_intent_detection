import pickle
import keras
from keras.preprocessing.sequence import pad_sequences
from model import *
import pandas as pd
import numpy as np
import time
start_time = time.time()
from csv import writer

def get_tokenizer():
  # loading
  with open('/content/drive/MyDrive/Fyp-Prototype/Unseen_intent_detection/tokenizer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  return tokenizer

def make_tokens(text, tokenizer):
  sequences = tokenizer.texts_to_sequences(text)
  sequences_pad = pad_sequences(sequences, maxlen=None, padding='post', truncating='post')
  return sequences_pad

def get_intent_labels(y, y_lof, labels):
  pred_labels = []
  y_df = pd.DataFrame(y, columns=labels)
  row = y_df.iloc[0]
  for label in labels:
    if row[label] == 1.0:
        pred_labels.append(label)
  if y_lof ==-1 :
    pred_labels.append('unseen')
  return pred_labels

def make_prediction(text):
    print(pickle.format_version)
    # loading models
    with open('Models/ohe.pickle', 'rb') as handle:
        ohe_new = pickle.load(handle)
    with open('Models/tokenizer.pickle',
              'rb') as handle:
        tokenizer_new = pickle.load(handle)
    main_model_new = keras.models.load_model(
        'Models/main_model.h5',
        custom_objects={'lmcl': lmcl})
    get_deep_feature_new = keras.models.load_model(
        'Models/get_deep_feature.h5')
    with open('Models/lof.pickle', 'rb') as handle:
        lof_new = pickle.load(handle)

    sequences_pad_test = make_tokens([text], tokenizer_new)
    x = sequences_pad_test

    y_proba = main_model_new.predict(x)

    y_proba = np.where((y_proba > 0.58), 1, y_proba)
    y_proba = np.where((y_proba < 0.58), 0, y_proba)

    x_deep_features = get_deep_feature_new.predict(x)

    y_lof = lof_new.predict(x_deep_features)

    pred_labels = get_intent_labels(y_proba, y_lof, ohe_new.classes_)
    print(pred_labels)
    print("--- %s seconds ---" % (time.time() - start_time))
    if 'unseen' in pred_labels:
        row_content = [text,pred_labels]
        append_list_as_row('report.csv', row_content)
    return pred_labels

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)