"""Métricas de evaluación a nivel píxel."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)


def flatten(items, preds, max_pixels=None, seed=42):
    """Concatena y (verdad), score y predicción de todas las imágenes."""
    y = np.concatenate([it["mask"].ravel() for it in items]).astype(np.uint8)
    s = np.concatenate([p["score"].ravel() for p in preds]).astype(np.float32)
    yp = np.concatenate([p["pred"].ravel() for p in preds]).astype(np.uint8)
    if max_pixels and len(y) > max_pixels:
        idx = np.random.default_rng(seed).choice(len(y), max_pixels, replace=False)
        return y[idx], s[idx], yp[idx]
    return y, s, yp


def recall_at_precision(y, s, min_precision=0.80):
    """Máximo recall alcanzable con precisión >= min_precision (y el umbral que lo logra)."""
    p, r, t = precision_recall_curve(y, s)
    ok = np.flatnonzero(p[:-1] >= min_precision)
    if len(ok) == 0:
        return dict(recall=0.0, precision=float("nan"), threshold=float("nan"), alcanzable=False)
    i = ok[np.argmax(r[ok])]
    return dict(recall=float(r[i]), precision=float(p[i]), threshold=float(t[i]), alcanzable=True)


def confusion(y_true, y_pred, class_names=("fondo", "miotubo")):
    """Matriz de confusión K×K (filas = verdad, columnas = predicción), en conteos y normalizada."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    idx = [f"real: {c}" for c in class_names]
    cols = [f"pred: {c}" for c in class_names]
    counts = pd.DataFrame(cm, index=idx, columns=cols)
    norm = pd.DataFrame(cm / cm.sum(axis=1, keepdims=True).clip(min=1), index=idx, columns=cols)
    return counts, norm


def class_metrics(cm_counts, class_names=("fondo", "miotubo")):
    """IoU, Dice, precisión y recall por clase a partir de la matriz de confusión (sirve para K clases)."""
    cm = cm_counts.to_numpy().astype(float)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    df = pd.DataFrame({
        "IoU": tp / np.maximum(tp + fp + fn, 1),
        "Dice/F1": 2 * tp / np.maximum(2 * tp + fp + fn, 1),
        "precision": tp / np.maximum(tp + fp, 1),
        "recall": tp / np.maximum(tp + fn, 1),
        "soporte_px": cm.sum(1).astype(int),
    }, index=list(class_names))
    return df


def evaluate(items, preds, min_precision=0.80, class_names=("fondo", "miotubo"), max_pixels=None):
    y, s, yp = flatten(items, preds, max_pixels)
    counts, norm = confusion(y, yp, class_names)
    cls = class_metrics(counts, class_names)
    rap = recall_at_precision(y, s, min_precision)
    summary = {
        "PR-AUC (AP)": float(average_precision_score(y, s)),
        "ROC-AUC": float(roc_auc_score(y, s)),
        f"recall@P>={min_precision:.2f}": rap["recall"],
        "umbral@P": rap["threshold"],
        "PR-AUC azar (prevalencia)": float(y.mean()),
        "IoU miotubo": float(cls.loc[class_names[1], "IoU"]),
        "Dice miotubo": float(cls.loc[class_names[1], "Dice/F1"]),
        "precision miotubo": float(cls.loc[class_names[1], "precision"]),
        "recall miotubo": float(cls.loc[class_names[1], "recall"]),
        "mIoU": float(cls["IoU"].mean()),
    }
    curves = dict(pr=precision_recall_curve(y, s), roc=roc_curve(y, s))
    return dict(summary=summary, confusion=counts, confusion_norm=norm, per_class=cls,
                recall_at_precision=rap, curves=curves)


def per_image(items, preds):
    rows = []
    for it, p in zip(items, preds):
        y, yp = it["mask"].astype(bool), p["pred"].astype(bool)
        tp = (y & yp).sum(); fp = (~y & yp).sum(); fn = (y & ~yp).sum()
        rows.append(dict(stem=it["stem"], frac_fg=y.mean(),
                         IoU=tp / max(tp + fp + fn, 1), Dice=2 * tp / max(2 * tp + fp + fn, 1),
                         precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1)))
    return pd.DataFrame(rows)
