import data
from model import load_model


def write_to_file(d: dict):
    with open("validation.txt", "w") as f:
        res = "\n".join([f"{k}: {round(d[k], 3)}" for k in
                         ["True positive", 'False positive', 'True negative', 'False negative', 'Accuracy', 'Precision',
                          'Recall']])
        precision, recall = d['Precision'], d['Recall']
        f1 = (precision * recall * 2) / (precision + recall)
        res += f"\nF1: {round(f1, 3)}\n"
        f.write(res)


if __name__ == '__main__':
    tk = data.learning_tk(d="data/")
    x_val, y_val = data.get_data(tk, status="val", d="data/")

    model = load_model()
    d = model.evaluate(x_val, y_val, batch_size=50, return_dict=True)

    write_to_file(d)
