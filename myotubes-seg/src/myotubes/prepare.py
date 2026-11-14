"""Paso 3 del pipeline: escribe los conjuntos de entrenamiento/validación/prueba listos para usar.

Lee los CSV de partición, redimensiona imagen (INTER_AREA) y máscara (vecino más cercano)
y escribe data/processed/sets/<split>/{images,masks}/<stem>.png + <split>.csv.
Los originales no se modifican.
"""
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .io import imwrite, read_gray, read_mask, resize


def prepare_sets(raw_root, out_dir, splits=("train", "val", "test"), scale=0.5):
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    written = {}
    for s in splits:
        csv = out_dir / "splits" / f"{s}.csv"
        df = pd.read_csv(csv)
        rows = []
        for r in tqdm(df.itertuples(index=False), total=len(df), desc=f"preparando {s}"):
            img = resize(read_gray(raw_root / r.image), scale)
            m = resize(read_mask(raw_root / r.mask), scale, is_mask=True)
            ip = out_dir / "sets" / s / "images" / f"{r.stem}.png"
            mp = out_dir / "sets" / s / "masks" / f"{r.stem}.png"
            imwrite(ip, img)
            imwrite(mp, m)          # valores 0/1
            rows.append(dict(stem=r.stem, image=ip.relative_to(out_dir).as_posix(),
                             mask=mp.relative_to(out_dir).as_posix(), group=r.group,
                             height=img.shape[0], width=img.shape[1], frac_fg=float(m.mean())))
        out = pd.DataFrame(rows)
        out.to_csv(out_dir / "sets" / f"{s}.csv", index=False)
        written[s] = out
    return written
