import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import xgboost as xgb


def matric_cf(y_test, y_pred):
    # Creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualization of the confusion matrix
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()

    classes = ["Class 0", "Class 1"]
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel("Predicated")
    plt.ylabel("True")

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2.0 else "black",
            )

    plt.show()


def roc_dg(model, X_test, y_test):

    # Calculating the probabilities of belonging to a positive class
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except:  # Condition added for XGBoost model
        y_prob = model.predict(dtest, output_margin=True)
        # y_prob = model.predict(dtest)
        y_prob = 1.0 / (
            1.0 + np.exp(-y_prob)
        )  # Apply sigmoid function to convert to probability
        y_prob = y_prob[:, 1]

    # Calculating the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Calculation of area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # ROC curve graph
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (AUC = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.show()
