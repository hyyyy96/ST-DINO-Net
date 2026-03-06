import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch


def compute_metrics(all_labels, all_preds, class_names=None):
    """
    Compute classification metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    if class_names:
        report = classification_report(all_labels, all_preds,
                                       target_names=class_names, digits=4)
        metrics['report'] = report

    return metrics


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path=None, dpi=600):
    """
    Plot and save confusion matrix
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10

    cm = confusion_matrix(all_labels, all_preds)

    # Size conversion: 10cm width
    cm_to_inch = 1 / 2.54
    width = 10 * cm_to_inch
    height = 8 * cm_to_inch

    plt.figure(figsize=(width, height))

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"size": 10},
                     cbar_kws={"shrink": 0.8})

    plt.xlabel('Predicted Label', fontsize=10, fontname='Times New Roman')
    plt.ylabel('True Label', fontsize=10, fontname='Times New Roman')
    plt.title('Confusion Matrix', fontsize=10, fontname='Times New Roman', pad=10)

    # Format labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                       rotation_mode='anchor', fontname='Times New Roman', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center',
                       fontname='Times New Roman', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()