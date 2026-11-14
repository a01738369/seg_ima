"""Paso 1 del pipeline: inventario de los datos originales -> manifest.csv.

Una fila por imagen con rutas relativas, metadatos del nombre, estadísticas de la
máscara/etiquetas y un hash perceptual (dHash) para detectar duplicados.
No modifica ningún archivo original.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .io import IMG_EXT, list_files, read_gray, read_mask, parse_yolo, polygon_area_norm

RE_HASH = re.compile(r"^(?P<hash>[0-9a-fA-F]{8})-(?P<rest>.+)$")
RE_BEST = re.compile(
    r"^(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"_(?P<well>[A-Za-z]\d+)_(?P<f1>\d+)_(?P<f2>\d+)_(?P<tag>\w+)$"
)
RE_CELL = re.compile(r"^(?P<cell>cell\d+)$", re.IGNORECASE)


def parse_name(stem):
    """Separa el nombre en sus partes. Ej.: 'ab12cd34-<uuid>_A3_04_03_Best' o 'ab12cd34-cell21'."""
    out = dict(file_hash=None, uuid=None, well=None, field=None, cell_id=None, name_family="otro")
    m = RE_HASH.match(stem)
    rest = stem
    if m:
        out["file_hash"], rest = m["hash"], m["rest"]
    if (b := RE_BEST.match(rest)):
        out.update(uuid=b["uuid"].lower(), well=b["well"].upper(), field=f"{b['f1']}_{b['f2']}",
                   name_family="best")
    elif (c := RE_CELL.match(rest)):
        out.update(cell_id=c["cell"].lower(), name_family="cell")
    return out


def dhash_hex(gray, size=8):
    """dHash de 64 bits como texto hexadecimal (16 caracteres)."""
    import cv2
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, [1, 99])
    g = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    small = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return "".join(f"{int(''.join('1' if b else '0' for b in bits[i:i + 4]), 2):x}" for i in range(0, len(bits), 4))


def build_manifest(raw_root, tiny_area_px=200):
    raw_root = Path(raw_root)
    imgs = list_files(raw_root / "images", IMG_EXT)
    masks = list_files(raw_root / "masks_png", IMG_EXT)
    labels = list_files(raw_root / "labels", {".txt"})
    vers = list_files(raw_root / "verification", IMG_EXT)

    rows = []
    for stem, ip in tqdm(imgs.items(), desc="manifiesto"):
        g = read_gray(ip)
        H, W = g.shape
        row = dict(stem=stem, image=ip.relative_to(raw_root).as_posix(),
                   mask=masks[stem].relative_to(raw_root).as_posix() if stem in masks else None,
                   label=labels[stem].relative_to(raw_root).as_posix() if stem in labels else None,
                   verification=vers[stem].relative_to(raw_root).as_posix() if stem in vers else None,
                   height=H, width=W, dtype=str(g.dtype),
                   int_mean=float(g.mean()), int_std=float(g.std()),
                   dhash=dhash_hex(g))
        row.update(parse_name(stem))

        if stem in masks:
            m = read_mask(masks[stem])
            row["px_fg"] = int(m.sum())
            row["frac_fg"] = float(m.mean())
        else:
            row["px_fg"], row["frac_fg"] = np.nan, np.nan

        n_inst = n_oor = n_tiny = 0
        max_frac = 0.0
        if stem in labels:
            polys, issues = parse_yolo(labels[stem])
            n_inst = len(polys)
            n_oor = sum(1 for _, k in issues if k == "fuera_de_rango")
            for _, pts in polys:
                a = polygon_area_norm(pts)
                max_frac = max(max_frac, a)
                n_tiny += int(a * H * W < tiny_area_px)
        row.update(n_inst=n_inst, n_out_of_range=n_oor, n_tiny=n_tiny, max_inst_frac=max_frac)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["empty_mask"] = df["px_fg"].fillna(0) == 0
    return df


def save_manifest(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.csv"
    df.to_csv(path, index=False)
    return path
