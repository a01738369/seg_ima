"""Líneas base sin deep learning (sección 4.4 del documento).

A) Clásica: normalización -> filtro de crestas oscuras (Sato) -> umbral elegido en train
   -> morfología + filtro por área.
B) Clasificador de píxeles: características hechas a mano + HistGradientBoosting.

Ambas producen un *score* continuo por píxel (para PR-AUC / ROC-AUC) y una máscara binaria.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sato
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from tqdm.auto import tqdm

from .io import read_gray, read_mask, resize


def load_items(df, root, scale):
    """Lista de dicts {stem, img (uint8), mask (0/1)} a la escala dada."""
    root = Path(root)
    items = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="cargando"):
        items.append(dict(stem=r.stem,
                          img=resize(read_gray(root / r.image), scale),
                          mask=resize(read_mask(root / r.mask), scale, is_mask=True)))
    return items


def normalize(img):
    g = img.astype(np.float32)
    lo, hi = np.percentile(g, [1, 99])
    return np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)


def ridge(img_norm, sigmas):
    """Respuesta de crestas OSCURAS (los miotubos son más oscuros que el fondo)."""
    return sato(img_norm, sigmas=sigmas, black_ridges=True).astype(np.float32)


def best_f1_threshold(y, s, prevalence=None):
    """Umbral que maximiza F1. Si la muestra está balanceada, se re-pondera a la prevalencia real."""
    w = None
    if prevalence is not None:
        w = np.where(y == 1, prevalence / y.mean(), (1 - prevalence) / (1 - y.mean()))
    p, r, t = precision_recall_curve(y, s, sample_weight=w)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i]), float(f1[i])


def _sample(y, rng, n_per_class):
    idx = []
    for c in (0, 1):
        pos = np.flatnonzero(y == c)
        if len(pos):
            idx.append(rng.choice(pos, min(n_per_class, len(pos)), replace=False))
    return np.concatenate(idx)


class ClassicBaseline:
    name = "Clásica (Sato + umbral + morfología)"

    def __init__(self, sigmas=(2, 3, 4, 5), min_object_px=30, seed=42):
        self.sigmas, self.min_object_px, self.seed = list(sigmas), min_object_px, seed
        self.scale_ = None
        self.threshold_ = None

    def _raw(self, img):
        return ridge(normalize(img), self.sigmas)

    def fit(self, items, n_per_class=150_000):
        """Calcula la escala del score (p99.5 en train) y el umbral que maximiza F1 en train."""
        rng = np.random.default_rng(self.seed)
        per_img = max(1, n_per_class // len(items))
        ref, S, Y = [], [], []
        npos = ntot = 0
        for it in tqdm(items, desc="Sato train"):
            r = self._raw(it["img"]).ravel(); y = it["mask"].ravel()
            ref.append(rng.choice(r, min(20_000, r.size), replace=False))
            idx = _sample(y, rng, per_img)
            S.append(r[idx]); Y.append(y[idx])
            npos += int(y.sum()); ntot += y.size
        self.scale_ = float(np.percentile(np.concatenate(ref), 99.5)) + 1e-12
        S = np.clip(np.concatenate(S) / self.scale_, 0, 1); Y = np.concatenate(Y)
        self.threshold_, self.train_f1_ = best_f1_threshold(Y, S, npos / ntot)
        return self

    def score(self, img):
        return np.clip(self._raw(img) / self.scale_, 0, 1)

    def predict_mask(self, score):
        return postprocess(score >= self.threshold_, self.min_object_px)


def postprocess(m, min_object_px):
    """Apertura 3x3 + elimina componentes conexas con menos de min_object_px píxeles."""
    m = ndi.binary_opening(m, structure=np.ones((3, 3)))
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_object_px
        keep[0] = False
        m = keep[lab]
    return m.astype(np.uint8)


def pixel_features(img, sigmas):
    n = normalize(img)
    g1 = cv2.GaussianBlur(n, (0, 0), 1.5)
    g2 = cv2.GaussianBlur(n, (0, 0), 4)
    mean = cv2.blur(n, (9, 9)); sq = cv2.blur(n * n, (9, 9))
    lstd = np.sqrt(np.clip(sq - mean ** 2, 0, None))
    bg = cv2.GaussianBlur(n, (0, 0), 25)
    dark = bg - g1                       # qué tan oscuro es el píxel respecto a su fondo local
    rd = ridge(n, sigmas)
    return np.stack([n, g1, g2, lstd, dark, rd], axis=-1).reshape(-1, 6)


FEATURE_NAMES = ["intensidad", "gauss_1.5", "gauss_4", "std_local_9", "oscuridad_vs_fondo", "sato"]


class PixelClassifierBaseline:
    name = "Clasificador de píxeles (características + HGB)"

    def __init__(self, sigmas=(2, 3, 4, 5), min_object_px=30, seed=42):
        self.sigmas, self.min_object_px, self.seed = list(sigmas), min_object_px, seed
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                  random_state=seed)

    def fit(self, items, n_per_class=150_000):
        rng = np.random.default_rng(self.seed)
        per_img = max(1, n_per_class // len(items))
        X, Y = [], []
        for it in tqdm(items, desc="características train"):
            f = pixel_features(it["img"], self.sigmas); y = it["mask"].ravel()
            idx = _sample(y, rng, per_img)
            X.append(f[idx]); Y.append(y[idx])
        X, Y = np.concatenate(X), np.concatenate(Y)
        self.clf.fit(X, Y)
        # Umbral que maximiza F1 en train (sin tocar validación). Muestra balanceada ->
        # se re-pondera para reflejar la prevalencia real de train.
        prev = sum(int(it["mask"].sum()) for it in items) / sum(it["mask"].size for it in items)
        p = self.clf.predict_proba(X)[:, 1]
        self.threshold_, self.train_f1_ = best_f1_threshold(Y, p, prev)
        return self

    def score(self, img):
        p = self.clf.predict_proba(pixel_features(img, self.sigmas))[:, 1]
        return p.reshape(img.shape).astype(np.float32)

    def predict_mask(self, score):
        return postprocess(score >= self.threshold_, self.min_object_px)


def predict_items(model, items):
    out = []
    for it in tqdm(items, desc="predicción"):
        s = model.score(it["img"])
        out.append(dict(stem=it["stem"], score=s, pred=model.predict_mask(s)))
    return out
