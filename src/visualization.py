import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.preprocessing import label_binarize


# Distribution plots for each numeric column
def plot_distributions(data):
    numeric_cols = data.select_dtypes(include='number').columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()


# Correlation heatmap
def plot_correlation_heatmap(data):
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()


def plot_pairplot(data):
    sns.pairplot(data, hue='quality', palette='Purples', corner=True)
    plt.show()


# Confusion Matrix and Classification Report
def evaluate_model(y_test, y_pred, model_name):
    print(f"Classification Report for {model_name}:\n",
          classification_report(y_test,
                                y_pred,
                                target_names=["Bad", "Middle", "Good"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Bad", "Middle", "Good"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


def plot_roc_curve(y_test, y_score, model_name):
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=["Bad", "Middle", "Good"])
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i+1} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.show()
