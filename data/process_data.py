import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

MAX_COL_LETTERS = 52
MAX_COL_WORDS = 126
BATCH = 64


def clear_data(data: pd.DataFrame):
    data["libs"] = (data["libs"].str.replace("/", "").str.replace("\\", "").str.replace("%", "").
                    str.replace("+", "").str.replace(":", "").str.replace(" ", "")
                    .str.split(","))
    return data


def learning_tk(d="", files=["train.tsv", "test.tsv", "val.tsv"]):
    # global MAX_COL_LETTERS, MAX_COL_WORDS

    l_libs = []

    for name in files:
        data = pd.read_csv(d + name, sep="\t")

        data = clear_data(data)

        for l in data.libs:
            l_libs.extend(l)
            # MAX_COL_WORDS = len(l) if MAX_COL_WORDS < len(l) else MAX_COL_WORDS # 126

    # MAX_COL_LETTERS = len(max(l_libs, key=len))  # 52

    tk = Tokenizer(
        filters='!"#$%&()*+,/:;<=>?@[\\]^`{|}~\t\n',
        char_level=True
    )

    tk.fit_on_texts(l_libs)
    return tk


def positional_encoding(length=10, depth=10):
    depth = depth // 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return pos_encoding.astype(float)


def pad_letters(tk, data: str):
    global MAX_COL_LETTERS
    return tk.texts_to_matrix(" " * (MAX_COL_LETTERS - len(data)) + data)


def pad_words(tk, data: list[str]):
    global MAX_COL_WORDS, MAX_COL_LETTERS
    return np.array(
        [pad_letters(tk, " ") * MAX_COL_LETTERS] * (MAX_COL_WORDS - len(data)) + [pad_letters(tk, i) for i in data])


def get_data(tk, status="train", d=""):
    data = clear_data(pd.read_csv(d + status + ".tsv", sep="\t"))
    x = np.array([pad_words(tk, libs) for libs in data["libs"].to_list()], dtype=float)

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
        X = np.array([pad_words(tk, libs) for libs in subset.libs.to_list()], dtype=float)

        if status == "test":
            yield X
        else:
            Y = np.array(subset["is_virus"], dtype=float).reshape(-1, 1)
            yield X, Y

        n += BATCH
        l += BATCH


if __name__ == '__main__':
    # tk = learning_tk()

    print(next(generator_data(tk))[0].shape)  # 32 126 52 40
