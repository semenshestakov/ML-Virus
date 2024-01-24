from tensorflow.keras.layers import *
import tensorflow as tf
from model.metrics import metrics
from data import MAX_CODE_WORDS, MAX_COL_WORDS

def load_model(name: str = "model_best"):
    model = tf.keras.models.load_model(name)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=metrics
    )
    return model

def get_model_for_lib(l=128, dp=0.2):
    global MAX_COL_WORDS, MAX_CODE_WORDS

    inp = Input((MAX_COL_WORDS, MAX_CODE_WORDS))  # 126 2048
    x = Dense(l, "relu", name="Dense_lib_1")(inp)
    x = BatchNormalization()(x)
    x = Dense(l // 4, "relu", name="Dense_lib_2")(x)
    x = BatchNormalization()(x)
    x = Dense(1, "relu", name="output_lib")(x)
    out = Reshape((126,), name="Reshape_layer")(x)

    return tf.keras.Model(inp, out)  # -> 126, 52


def get_model_for_file(l=128, dp=0.2):
    inp = Input((MAX_COL_WORDS,))  # 126
    x = Dropout(dp)(BatchNormalization()(inp))
    x = Dense(l, "relu", name="Dense_file_1")(x)
    x = Dropout(dp)(BatchNormalization()(x))
    x = Dense(l // 4, "relu", name="Dense_file_2")(x)
    out = Dense(1, "relu", name="output_file")(x)

    return tf.keras.Model(inp, out)

@tf.function
def train_on_step(model: tf.keras.Model, x, y,sw):

    with tf.GradientTape() as tape:
        loss = model.loss(y, model(x), sample_weight=sw)

    grad = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grad, model.trainable_weights))


if __name__ == '__main__':
    print(get_model_for_lib().summary())

    print(get_model_for_file().summary())
    # print(positional_encoding(MAX_COL_LETTERS,40).shape)
