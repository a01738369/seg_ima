"""Pipeline de datos: originales -> manifest.csv -> train/val/test.csv -> conjuntos preparados.

Uso (desde la raíz del repositorio):
    python scripts/run_pipeline.py --config config.json
    python scripts/run_pipeline.py --config config.json --raw "C:\\ruta\\a\\vt" --steps manifest split
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from myotubes import load_config  # noqa: E402
from myotubes.io import find_root  # noqa: E402
from myotubes.manifest import build_manifest, save_manifest  # noqa: E402
from myotubes.split import run_split  # noqa: E402
from myotubes.prepare import prepare_sets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--raw", help="Ruta a la carpeta vt (sobrescribe raw_dir del config)")
    ap.add_argument("--steps", nargs="+", default=["manifest", "split", "prepare"],
                    choices=["manifest", "split", "prepare"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = Path(args.raw or cfg["raw_dir"])
    raw = raw if raw.is_absolute() else ROOT / raw
    raw_root = find_root(raw)
    out_dir = Path(cfg["out_dir"])
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    print("Datos originales:", raw_root)
    print("Salida:", out_dir)

    if "manifest" in args.steps:
        df = build_manifest(raw_root)
        print("manifest.csv ->", save_manifest(df, out_dir), f"({len(df)} imágenes)")

    if "split" in args.steps:
        man = pd.read_csv(out_dir / "manifest.csv")
        df, pairs, info = run_split(man, cfg, out_dir)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        print(df["split"].value_counts().to_string())

    if "prepare" in args.steps:
        sets = prepare_sets(raw_root, out_dir, tuple(cfg["split"]), cfg["prepare"]["scale"])
        for k, v in sets.items():
            print(f"{k}: {len(v)} imágenes -> {out_dir / 'sets' / k}")


if __name__ == "__main__":
    main()
