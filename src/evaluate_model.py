"""
evaluate_model.py
-------------------
Produces the evaluation report that actually matters for fraud detection:
confusion matrix, precision/recall/F1, ROC-AUC, PR-AUC, and plots.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import config


def evaluate_model(model, X_test, y_test, threshold: float = config.CLASSIFICATION_THRESHOLD):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=4)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "y_proba": y_proba,
        "y_pred": y_pred,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "report": report,
    }


def plot_confusion_matrix(cm, save_path: str = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"]).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    plt.close(fig)


def plot_roc_curve(y_test, y_proba, save_path: str = None):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    plt.close(fig)


def plot_precision_recall_curve(y_test, y_proba, save_path: str = None):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    plt.close(fig)


def generate_all_plots(results, y_test, output_dir: str = None):
    output_dir = output_dir or os.path.join(config.ROOT_DIR, "models", "plots")
    os.makedirs(output_dir, exist_ok=True)

    plot_confusion_matrix(results["confusion_matrix"], os.path.join(output_dir, "confusion_matrix.png"))
    plot_roc_curve(y_test, results["y_proba"], os.path.join(output_dir, "roc_curve.png"))
    plot_precision_recall_curve(y_test, results["y_proba"], os.path.join(output_dir, "pr_curve.png"))
