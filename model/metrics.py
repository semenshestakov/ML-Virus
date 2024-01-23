from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from tensorflow.keras.metrics import Accuracy, Precision, Recall, BinaryAccuracy

metrics = [
    TruePositives(name="True positive"),
    FalsePositives(name="False positive"),
    TrueNegatives(name="True negative"),
    FalseNegatives(name="False negative"),
    BinaryAccuracy(name="Accuracy"),
    Precision(name="Precision"),
    Recall(name="Recall"),
]


def print_metrics(metrics:dict[str]):
    print(*[f"{m}: {metrics[m]}" for m in metrics],sep="\n")