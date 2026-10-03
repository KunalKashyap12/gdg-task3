"""
Evaluation and diagnostic module for News Classification System.
Computes metrics, plots confusion matrix, extracts top keywords per class,
and performs error analysis.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.config import LABEL_MAP, CLASS_NAMES, CLASS_COLORS


def calculate_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Computes overall and per-class classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_p = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            cls_name: {
                "precision": float(clf_report[cls_name]["precision"]),
                "recall": float(clf_report[cls_name]["recall"]),
                "f1_score": float(clf_report[cls_name]["f1-score"]),
                "support": int(clf_report[cls_name]["support"])
            }
            for cls_name in CLASS_NAMES
        },
        "confusion_matrix": cm.tolist()
    }
    return metrics


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path: Optional[Path] = None,
    title: str = "News Classification Confusion Matrix"
):
    """
    Generates and saves a publication-quality confusion matrix heatmap
    with both counts and percentages.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Annotations: Count + Percentage
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Normalized Recall"},
        linewidths=1.2,
        linecolor="#E2E8F0",
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Category", fontsize=12, fontweight="semibold", labelpad=10)
    ax.set_ylabel("True Category", fontsize=12, fontweight="semibold", labelpad=10)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix plot saved to {save_path}")

    return fig, ax


def plot_per_class_metrics(
    metrics_dict: Dict[str, Any],
    save_path: Optional[Path] = None
):
    """
    Plots a multi-bar chart comparing Precision, Recall, and F1-Score per class.
    """
    per_class = metrics_dict["per_class"]
    classes = list(per_class.keys())
    precisions = [per_class[c]["precision"] * 100 for c in classes]
    recalls = [per_class[c]["recall"] * 100 for c in classes]
    f1s = [per_class[c]["f1_score"] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects1 = ax.bar(x - width, precisions, width, label="Precision", color="#3B82F6", alpha=0.9)
    rects2 = ax.bar(x, recalls, width, label="Recall", color="#10B981", alpha=0.9)
    rects3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#8B5CF6", alpha=0.9)

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="semibold")
    ax.set_title("Performance Metrics by News Category", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11, fontweight="semibold")
    ax.set_ylim(80, 100)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add value labels above bars
    for rect in rects1 + rects2 + rects3:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Per-class metrics plot saved to {save_path}")

    return fig, ax


def get_top_features_per_class(
    vectorizer,
    classifier,
    top_n: int = 15,
    save_path: Optional[Path] = None
) -> Dict[str, List[str]]:
    """
    Extracts and visualizes the most informative features / n-grams per news category.
    Works for LinearSVC, LogisticRegression, or calibrated wrappers with base linear estimator.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Extract coefficients
    if hasattr(classifier, "coef_"):
        coef = classifier.coef_
    elif hasattr(classifier, "calibrated_classifiers_"):
        # Average coefficients across calibrated folds
        coefs = [clf.estimator.coef_ for clf in classifier.calibrated_classifiers_]
        coef = np.mean(coefs, axis=0)
    else:
        return {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    top_features = {}
    for idx, cls_name in enumerate(CLASS_NAMES):
        top_indices = np.argsort(coef[idx])[-top_n:]
        top_words = feature_names[top_indices]
        top_weights = coef[idx][top_indices]
        top_features[cls_name] = list(reversed(top_words.tolist()))

        ax = axes[idx]
        ax.barh(top_words, top_weights, color=CLASS_COLORS[cls_name], alpha=0.85)
        ax.set_title(f"Top Predictive Features: {cls_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Model Coefficient Weight", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Top keywords plot saved to {save_path}")

    return top_features


def perform_error_analysis(
    df_test: pd.DataFrame,
    y_true,
    y_pred,
    max_examples_per_pair: int = 2
) -> List[Dict[str, Any]]:
    """
    Identifies common confusion pairs and extracts concrete misclassified samples.
    """
    errors = []
    for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label != pred_label:
            errors.append({
                "index": idx,
                "title": df_test.iloc[idx]["Title"],
                "description": df_test.iloc[idx]["Description"],
                "true_id": int(true_label),
                "true_category": LABEL_MAP[true_label],
                "pred_id": int(pred_label),
                "pred_category": LABEL_MAP[pred_label]
            })

    error_df = pd.DataFrame(errors)
    confusion_counts = error_df.groupby(["true_category", "pred_category"]).size().reset_index(name="count")
    confusion_counts = confusion_counts.sort_values(by="count", ascending=False)

    sample_errors = []
    for _, row in confusion_counts.head(5).iterrows():
        tc, pc, cnt = row["true_category"], row["pred_category"], row["count"]
        subset = error_df[(error_df["true_category"] == tc) & (error_df["pred_category"] == pc)]
        examples = subset.head(max_examples_per_pair).to_dict(orient="records")
        sample_errors.append({
            "pair": f"True: {tc} -> Predicted: {pc}",
            "misclassified_count": int(cnt),
            "examples": examples
        })

    return sample_errors
