import numpy as np
from tensorflow.keras.layers import *
from data import *
import tensorflow as tf




def get_model_for_letters(l = 128,dp=0.3):
    global MAX_COL_WORDS, MAX_COL_LETTERS

    inp = Input((process_data.MAX_COL_WORDS, process_data.MAX_COL_LETTERS, 40))
    x = Dense(256, "relu", name="Dense_lettters_1")(inp)
    x = Dropout(dp)(BatchNormalization()(x))
    x = Dense(l, "relu", name="Dense_lettters_2")(x)
    x = Dropout(dp)(BatchNormalization()(x))
    x = Dense(1, "relu", name="output")(x)
    out = Reshape((process_data.MAX_COL_WORDS, process_data.MAX_COL_LETTERS), name="Reshape_output")(x)

    return tf.keras.Model(inp, out)  # -> 126, 52


def get_model_for_words(l=50,dp=0.3):
    global MAX_COL_WORDS, MAX_COL_LETTERS

    inp = Input((process_data.MAX_COL_WORDS, process_data.MAX_COL_LETTERS))
    x = GRU(128, "relu", return_sequences=True, name="GRU_seq",dropout=dp/2)(inp)  # 126 30
    x = GRU(50, "relu", name="GRU",dropout=0.1)(x)  # 126 30
    x = Dropout(dp)(BatchNormalization()(x))
    x = Dense(l, "relu", name="Dense_word_1")(x)
    x = Dropout(dp)(BatchNormalization()(x))
    out = Dense(1, "sigmoid", name="output")(x)

    return tf.keras.Model(inp, out)  # -> 126, 52


if __name__ == '__main__':
    print(get_model_for_letters().summary())
    print(get_model_for_words().summary())
    # print(positional_encoding(MAX_COL_LETTERS,40).shape)
