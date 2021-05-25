# packages for model building
from keras import backend as K
from keras.models import Sequential
from keras.layers import *
from keras.constraints import unit_norm
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import layers
from tensorflow import keras
import pandas as pd

def lmcl(traget_true, traget_pred, scl=30, mrgn=0.35):
    traget_pred = (1 - traget_true) * traget_pred + traget_true * (traget_pred - mrgn)
    traget_pred *= scl
    return K.categorical_crossentropy(traget_true, traget_pred, from_logits=True)


def build_model(max_sq_len, max_features, emb_dim, out_dim, bilstm_lmcl_model_path=None, emb_matrix=None):
    text_inputs = Input(shape=(max_sq_len,))
    if emb_matrix is None:
        embedding_layer = layers.Embedding(max_features, emb_dim, input_length=max_sq_len, mask_zero=True)(text_inputs)
    else:
        embedding_layer = layers.Embedding(max_features, emb_dim, input_length=max_sq_len, mask_zero=True,
                                           weights=[emb_matrix], trainable=True)(text_inputs)

    biLSTM_layer = layers.Bidirectional(LSTM(256, dropout=0.5))(embedding_layer)
    dropout_1 = layers.Dropout(0.5)(biLSTM_layer)
    dence1 = layers.Dense(128,activation='relu')(dropout_1)
    intent_outputs = layers.Dense(out_dim, activation='sigmoid', use_bias=False, kernel_constraint=unit_norm())(dence1)
    adam = Adam(lr=0.003, clipnorm=5.)
    bilstm_lmcl_model = keras.Model(inputs=text_inputs, outputs=intent_outputs)
    bilstm_lmcl_model.compile(loss=lmcl, optimizer=adam, metrics=['accuracy'])

    if bilstm_lmcl_model_path:
        plot_model(bilstm_lmcl_model, to_file=bilstm_lmcl_model_path, show_shapes=True, show_layer_names=False)

    return bilstm_lmcl_model

data = pd.DataFrame(columns=["utterances", "Identified_intents"])
data.to_csv('report.csv', mode='w')