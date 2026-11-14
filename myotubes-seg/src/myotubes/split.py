"""Paso 2 del pipeline: exclusiones documentadas, deduplicación y partición agrupada."""
from pathlib import Path

import numpy as np
import pandas as pd

from .leakage import duplicate_pairs, duplicate_clusters, build_groups


def apply_exclusions(df, cfg):
    """Marca (no borra) las imágenes excluidas y el motivo."""
    ex = cfg["exclude"]
    reason = pd.Series("", index=df.index)
    if ex.get("missing_mask", True):
        reason[df["mask"].isna()] = "sin_mascara"
    if ex.get("empty_masks", True):
        reason[(reason == "") & df["empty_mask"]] = "mascara_vacia"
    thr = ex.get("max_instance_frac")
    if thr:
        reason[(reason == "") & (df["max_inst_frac"] > thr)] = f"caja_gigante(>{thr:.0%})"
    out = df.copy()
    out["excluded_reason"] = reason
    return out


def dedupe(df, clusters):
    """Deja una copia por clúster de duplicados: la de más píxeles anotados."""
    d = df.copy()
    d["dup_cluster"] = d["stem"].map(clusters)
    d["dup_size"] = d.groupby("dup_cluster")["stem"].transform("size")
    keep = (d.sort_values(["dup_cluster", "px_fg", "stem"], ascending=[True, False, True])
              .groupby("dup_cluster").head(1)["stem"])
    d["is_representative"] = d["stem"].isin(keep)
    return d


def group_split(groups, fracs, seed=42):
    """Asigna grupos completos a train/val/test acercándose a las fracciones por nº de imágenes."""
    sizes = groups.value_counts()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())
    # primero los grupos grandes (más difíciles de acomodar), en orden aleatorio estable
    order = sorted(order, key=lambda g: -sizes[g])
    total = sizes.sum()
    target = {k: v * total for k, v in fracs.items()}
    count = {k: 0 for k in fracs}
    assign = {}
    for g in order:
        k = max(fracs, key=lambda s: (target[s] - count[s]) / target[s])
        assign[g] = k
        count[k] += sizes[g]
    return groups.map(assign)


def run_split(manifest, cfg, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    df = manifest.copy()

    pairs, _ = duplicate_pairs(df, cfg["dhash_threshold"])
    clusters = duplicate_clusters(df, pairs)
    df = apply_exclusions(df, cfg)
    df = dedupe(df, clusters) if cfg.get("dedupe", True) else df.assign(
        dup_cluster=df["stem"].map(clusters), is_representative=True)

    usable = df[(df["excluded_reason"] == "") & df["is_representative"]].copy()
    groups, info = build_groups(df, pairs, use_uuid=cfg.get("group_by_uuid", True))
    usable["group"] = usable["stem"].map(groups)
    usable["split"] = group_split(usable.set_index("stem")["group"], cfg["split"],
                                  cfg["seed"]).reindex(usable["stem"]).to_numpy()

    # verificación: ningún grupo en dos particiones
    cross = usable.groupby("group")["split"].nunique()
    assert (cross == 1).all(), "Hay grupos repartidos en más de una partición"

    df = df.merge(usable[["stem", "group", "split"]], on="stem", how="left")
    df.loc[df["split"].isna() & (df["excluded_reason"] != ""), "split"] = "excluida"
    df.loc[df["split"].isna(), "split"] = "duplicado_descartado"

    df.to_csv(out_dir / "splits" / "all.csv", index=False)
    pairs.to_csv(out_dir / "splits" / "duplicate_pairs.csv", index=False)
    for s in cfg["split"]:
        df[df["split"] == s].to_csv(out_dir / "splits" / f"{s}.csv", index=False)
    df[df["split"].isin(["excluida", "duplicado_descartado"])].to_csv(
        out_dir / "splits" / "descartadas.csv", index=False)
    info["n_pares_duplicados"] = int(len(pairs))
    return df, pairs, info
