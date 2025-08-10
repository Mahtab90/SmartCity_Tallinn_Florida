# plotting_utils.py
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

def plot_roc_curves(curves, out_pdf="roc_curves.pdf"):
    """
    curves: list of dicts with keys: fpr, tpr, auc, label
    Saves a single-figure ROC plot (no specific colors).
    """
    fig = plt.figure(figsize=(6,5))
    ax = plt.gca()
    for c in curves:
        ax.plot(c["fpr"], c["tpr"], label=f'{c["label"]} (AUC={c["auc"]:.3f})')
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf",".png"), dpi=300, bbox_inches="tight")
    return out_pdf

def plot_feature_importance(importances, feature_names, out_pdf="feature_importance.pdf", top_k=20):
    """
    importances: array-like
    feature_names: list-like
    Saves a bar chart of top-k features by importance.
    """
    import numpy as np
    idx = np.argsort(importances)[::-1][:top_k]
    imp = [importances[i] for i in idx]
    names = [feature_names[i] for i in idx]
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.barh(range(len(imp))[::-1], imp[::-1])
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("XGBoost Feature Importance (Top-k)")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf",".png"), dpi=300, bbox_inches="tight")
    return out_pdf
