from data import *
import model as M

MAX_COL_WORDS = 126
MAX_CODE_WORDS = 2048
BATCH = 64


def train_model(epoch, d=""):
    tk = learning_tk(d=d)

    l_model = M.get_model_for_lib(128)  # 275137 None, 126,2048 -> None 126
    w_model = M.get_model_for_file(128)  # 16889 None 126 -> None 1

    inp = l_model.input
    x = l_model(inp)
    x = w_model(x)

    model = tf.keras.Model(inp, x)  # create full model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=M.metrics
    )

    t = 0
    acc, copy_model = 0., None
    x_val, y_val = get_data(tk, status="val", d=d)

    for e in range(1, epoch + 1):
        print(f"Epoch: {e}")

        gen = generator_data(tk, "train", d=d)

        for (x, y) in gen:
            sw = np.ones_like(y, dtype=float)
            sw[y == 1] = 0.25

            M.train_on_step(model, x, y, sw)

            if t % 50 == 0:

                d = model.evaluate(x_val, y_val, batch_size=50, return_dict=True)
                if d["Accuracy"] > acc:
                    print("New acc", acc)
                    acc = d["Accuracy"]
                    model.save("model_best")

                print()

            t += 1


if __name__ == '__main__':
    train_model(1, "data/")
