import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer


def main():
    dataframe = pd.read_csv("__files/sba_small.csv", header=0)
    dataframe = dataframe[['Selected', 'Recession', 'RealEstate', 'Portion', 'Default']]
    data = dataframe.values
    X = data[:, :3]
    y = data[:, 3]
    dataframe[dataframe.Selected == 1].to_csv("train.csv", index=False, header=True)
    dataframe[dataframe.Selected == 0].to_csv("test.csv", index=False, header=True)
    X_train = dataframe[dataframe.Selected == 1].drop(['Selected'], axis=1)
    X_test = dataframe[dataframe.Selected == 0].drop(['Selected'], axis=1)
    y_train = dataframe[dataframe.Selected == 1].Default
    y_test = dataframe[dataframe.Selected == 0].Default

    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    y_pred = logisticRegr.predict(X_test)
    s = joblib.dump(logisticRegr, "model.joblib")
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    with open('confusion_matrix.json', 'w') as output:
        data = {"tn": TN, "fp": FP, "fn": FN, "tp": TP}
        json.dump(data, output, sort_keys=False, default=str)

    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    macro_roc_auc_ovo = roc_auc_score(
        y_test,
        y_pred,
        multi_class="ovo",
        average="macro",
    )
    RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_pred.ravel(),
        name="micro-average OvR",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc.pdf")
    plt.show()


if __name__ == "__main__":
    main()
