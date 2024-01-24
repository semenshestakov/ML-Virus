import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import tensorflow as tf
from pprint import pprint

MAX_COL_WORDS = 126
MAX_CODE_WORDS = 2048
BATCH = 64


def clear_data(data: pd.DataFrame):
    data["libs"] = (data["libs"].str.replace("/", "").str.replace("\\", "").str.replace("%", "").
                    str.replace("+", "").str.replace(":", "").str.replace(" ", "")
                    .str.split(","))
    return data


def learning_tk(d="", files=["train.tsv", "test.tsv", "val.tsv"]):
    global MAX_COL_WORDS, MAX_CODE_WORDS

    l_libs = []

    for name in files:
        data = pd.read_csv(d + name, sep="\t")

        data = clear_data(data)

        for l in data.libs:
            l_libs.extend(l)

    tk = Tokenizer(
        filters='!"#$%&()*+,/:; <=>?@[\\]^`{|}~\t\n',
        char_level=False
    )

    tk.fit_on_texts(l_libs)
    return tk


def process_words(tk, data: list[str]):
    global MAX_COL_WORDS, MAX_CODE_WORDS
    return np.array(tk.texts_to_matrix( [""] * (MAX_COL_WORDS - len(data)) + data),dtype=float)



def get_data(tk, status="train", d=""):
    data = clear_data(pd.read_csv(d + status + ".tsv", sep="\t"))
    x = np.array([process_words(tk, libs) for libs in data["libs"].to_list()], dtype=float)

    if status == "test":
        return x

    y = np.array(data["is_virus"], dtype=float).reshape(-1, 1)
    return x, y


def generator_data(tk: Tokenizer, status="train",d=""):
    """
    :return: -> shape 52 126 40
    """
    data = clear_data(pd.read_csv(d + status + ".tsv", sep="\t"))

    if status == "train":
        data = data.sample(frac=1)

    l, n = 0, BATCH

    while n < len(data):
        subset = data.iloc[l:n]
        X = np.array([process_words(tk, libs) for libs in subset["libs"].to_list()], dtype=float)

        if status == "test":
            yield X
        else:
            Y = np.array(subset["is_virus"], dtype=float).reshape(-1, 1)
            yield X, Y

        n += BATCH
        l += BATCH




if __name__ == '__main__':
    tk = learning_tk()

    print(next(generator_data(tk))[0].shape)  #
    # pprint(np.array(tk.texts_to_matrix([""])).sum()) # 2048




