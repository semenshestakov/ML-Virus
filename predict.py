import data
from model import load_model


def write_to_file(x_pred_test):
    with open("prediction.txt", "w") as fp:
        fp.write("prediction\n")
        for i in x_pred_test:
            if i > 0.5:
                fp.write("1\n")
            else:
                fp.write("0\n")


if __name__ == '__main__':
    model = load_model()
    tk = data.learning_tk(d="data/")

    x_test = data.get_data(tk, status="test",d="data/")
    x_pred_test = model(x_test).numpy().reshape(-1, )

    write_to_file(x_pred_test)
