"""Detección de duplicados, construcción de grupos y diagnóstico de fuga de información."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_mask, resize


def hex_to_bits(h):
    return np.array([int(b) for c in h for b in f"{int(c, 16):04b}"], dtype=bool)


def hamming_matrix(hashes):
    bits = np.stack([hex_to_bits(h) for h in hashes])
    return (bits[:, None, :] != bits[None, :, :]).sum(-1)


def duplicate_pairs(df, threshold=6):
    """Pares (i, j, distancia) con distancia de Hamming del dHash <= threshold."""
    D = hamming_matrix(df["dhash"].tolist())
    iu = np.triu_indices(len(df), 1)
    mask = D[iu] <= threshold
    stems = df["stem"].to_numpy()
    pairs = pd.DataFrame({"stem_a": stems[iu[0][mask]], "stem_b": stems[iu[1][mask]],
                          "hamming": D[iu][mask]})
    return pairs.sort_values("hamming").reset_index(drop=True), D


class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _components(stems, edges):
    uf = UnionFind(stems)
    for a, b in edges:
        uf.union(a, b)
    roots = {s: uf.find(s) for s in stems}
    ids = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return pd.Series({s: ids[r] for s, r in roots.items()})


def _same_key_edges(df, col):
    edges = []
    for _, g in df.dropna(subset=[col]).groupby(col):
        s = g["stem"].tolist()
        edges += [(s[0], x) for x in s[1:]]
    return edges


def duplicate_clusters(df, pairs):
    """Clúster de copias: pares dHash + mismo cell_id."""
    edges = list(zip(pairs["stem_a"], pairs["stem_b"])) + _same_key_edges(df, "cell_id")
    return _components(df["stem"].tolist(), edges)


def build_groups(df, pairs, use_uuid=True, max_group_frac=0.35):
    """Grupos para la partición: duplicados + mismo cell_id (+ mismo UUID si use_uuid).

    Si agrupar por UUID produce un grupo con más de max_group_frac de las imágenes
    (los duplicados conectan lotes distintos), se descarta el UUID y se avisa.
    """
    base = list(zip(pairs["stem_a"], pairs["stem_b"])) + _same_key_edges(df, "cell_id")
    stems = df["stem"].tolist()
    info = {"modo": "duplicados + cell_id"}
    if use_uuid:
        g = _components(stems, base + _same_key_edges(df, "uuid"))
        biggest = g.value_counts().max() / len(g)
        if biggest <= max_group_frac:
            info = {"modo": "duplicados + cell_id + UUID", "grupo_mayor_frac": float(biggest)}
            return g, info
        info["aviso"] = (f"Agrupar por UUID genera un grupo con {biggest:.0%} de las imágenes; "
                         "se usa solo duplicados + cell_id.")
    g = _components(stems, base)
    info["grupo_mayor_frac"] = float(g.value_counts().max() / len(g))
    return g, info


def label_agreement(df, pairs, raw_root, scale=0.25, max_pairs=None):
    """IoU entre las máscaras de cada par duplicado (¿se anotaron igual?)."""
    raw_root = Path(raw_root)
    paths = df.set_index("stem")["mask"]
    rows = []
    sub = pairs if max_pairs is None else pairs.head(max_pairs)
    for a, b, h in sub[["stem_a", "stem_b", "hamming"]].itertuples(index=False):
        ma = resize(read_mask(raw_root / paths[a]), scale, True)
        mb = resize(read_mask(raw_root / paths[b]), scale, True)
        if ma.shape != mb.shape:
            iou = np.nan
        else:
            u = np.logical_or(ma, mb).sum()
            iou = 1.0 if u == 0 else np.logical_and(ma, mb).sum() / u
        rows.append(dict(stem_a=a, stem_b=b, hamming=h, iou_mascaras=iou))
    return pd.DataFrame(rows)


def leakage_rate(split_series, pairs, eval_split="val", ref_split="train"):
    """Fracción de imágenes de eval_split que tienen una copia (par dHash) en ref_split."""
    sp = split_series.to_dict()
    nbrs = defaultdict(set)
    for a, b in zip(pairs["stem_a"], pairs["stem_b"]):
        nbrs[a].add(b); nbrs[b].add(a)
    ev = [s for s, v in sp.items() if v == eval_split]
    if not ev:
        return np.nan
    leaked = sum(any(sp.get(n) == ref_split for n in nbrs[s]) for s in ev)
    return leaked / len(ev)


def random_split_leakage(stems, pairs, fracs, seeds=range(20)):
    """Simula particiones aleatorias ingenuas (sin grupos) y mide la fuga."""
    out = []
    stems = list(stems)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(stems)
        n_tr = round(fracs["train"] * len(perm)); n_va = round(fracs["val"] * len(perm))
        sp = pd.Series("test", index=perm)
        sp.iloc[:n_tr] = "train"; sp.iloc[n_tr:n_tr + n_va] = "val"
        out.append(leakage_rate(sp, pairs))
    return np.array(out)


def min_cross_distance(D, df, split_series, eval_split="val", ref_split="train"):
    """Para cada imagen de eval_split, distancia dHash mínima a cualquier imagen de ref_split."""
    idx = {s: i for i, s in enumerate(df["stem"])}
    ev = [idx[s] for s, v in split_series.items() if v == eval_split and s in idx]
    rf = [idx[s] for s, v in split_series.items() if v == ref_split and s in idx]
    if not ev or not rf:
        return np.array([])
    return D[np.ix_(ev, rf)].min(axis=1)
