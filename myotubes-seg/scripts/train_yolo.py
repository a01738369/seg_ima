"""Entrena YOLO (Ultralytics) sobre el dataset de miotubos usando la partición del pipeline.

Usa directamente las etiquetas YOLO-Seg de vt/labels/ (rectángulos rotados de 4 vértices) y los
CSV de partición agrupada de data/processed/splits/ (train.csv / val.csv), así que NO hace una
partición aleatoria propia y no hay fuga por imágenes duplicadas.

Tareas (--task):
  segment  (por defecto) segmentación por instancia con los polígonos tal cual  -> yolo11n-seg.pt
  obb      cajas orientadas; el formato de 4 vértices coincide con el de YOLO-OBB -> yolo11n-obb.pt
  detect   cajas alineadas a los ejes (bbox del polígono)                         -> yolo11n.pt

Ejemplos (desde la raíz del repositorio):
  python scripts/train_yolo.py --mode finetune
  python scripts/train_yolo.py --mode transfer --task obb --epochs 100
  python scripts/train_yolo.py --mode finetune --prepare-only     # solo escribe el dataset
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myotubes.io import find_root, parse_yolo  # noqa: E402

DEFAULT_WEIGHTS = {"segment": "yolo11n-seg.pt", "obb": "yolo11n-obb.pt", "detect": "yolo11n.pt"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("finetune", "transfer"), required=True)
    parser.add_argument("--task", choices=("segment", "obb", "detect"), default="segment")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "vt",
                        help="Carpeta vt original (acepta vt/ o vt/vt/)")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "processed" / "splits",
                        help="Carpeta con train.csv y val.csv generados por run_pipeline.py")
    parser.add_argument("--weights", default=None, help="Pesos preentrenados o checkpoint (por defecto según --task)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "yolo")
    parser.add_argument("--class-name", default="miotubo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Lado de entrada. 1024 deja el ancho típico (~40 px) en ~16 px")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--freeze-layers", type=int, default=10,
                        help="Capas iniciales congeladas en modo transfer; se ignora en finetune")
    parser.add_argument("--min-area-px", type=float, default=0,
                        help="Descarta polígonos con área menor (px² a resolución original)")
    parser.add_argument("--device", default=None, help="'0' para GPU, 'cpu', etc. (por defecto: automático)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true", help="Solo escribe el dataset y termina")
    return parser.parse_args()


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def load_split(splits_dir, name):
    csv = splits_dir / f"{name}.csv"
    if not csv.is_file():
        raise SystemExit(f"No existe {csv}. Corre antes: python scripts/run_pipeline.py --steps manifest split")
    df = pd.read_csv(csv)
    if df.empty:
        raise SystemExit(f"{csv} está vacío")
    return df


def polygon_area_px(pts, w, h):
    q = pts * [w, h]
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convert_label(label_path, task, w, h, min_area_px):
    """Convierte las líneas YOLO-Seg al formato de la tarea. Recorta coordenadas a [0,1]."""
    polys, _ = parse_yolo(label_path)
    lines, stats = [], {"clipped": 0, "dropped": 0}
    for _, pts in polys:
        if len(pts) < 3:
            stats["dropped"] += 1
            continue
        if (pts < 0).any() or (pts > 1).any():
            stats["clipped"] += 1
        pts = np.clip(pts, 0, 1)
        if polygon_area_px(pts, w, h) < max(min_area_px, 1e-9):
            stats["dropped"] += 1
            continue
        if task == "segment":
            coords = pts.ravel()
        elif task == "obb":
            if len(pts) != 4:      # YOLO-OBB requiere exactamente 4 vértices
                import cv2
                rect = cv2.minAreaRect((pts * [w, h]).astype(np.float32))
                pts = cv2.boxPoints(rect) / [w, h]
                pts = np.clip(pts, 0, 1)
            coords = pts.ravel()
        else:  # detect
            x0, y0 = pts.min(0); x1, y1 = pts.max(0)
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                stats["dropped"] += 1
                continue
            coords = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
        lines.append("0 " + " ".join(f"{v:.6f}" for v in coords))
    return lines, stats


def link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)          # hardlink: no duplica espacio en disco (mismo volumen)
    except OSError:
        shutil.copy2(src, dst)


def write_dataset(raw_root, splits_dir, output_dir, task, class_name, min_area_px):
    dataset_dir = output_dir / f"dataset_{task}"
    totals = {"clipped": 0, "dropped": 0, "instances": 0}
    for split in ("train", "val"):
        df = load_split(splits_dir, split)
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for r in df.itertuples(index=False):
            src = raw_root / r.image
            if not src.is_file():
                raise FileNotFoundError(f"No encontré la imagen {src}")
            dst = img_dir / f"{r.stem}{src.suffix.lower()}"
            link_or_copy(src, dst)
            lines, st = convert_label(raw_root / r.label, task, int(r.width), int(r.height), min_area_px)
            (lbl_dir / f"{r.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            totals["clipped"] += st["clipped"]; totals["dropped"] += st["dropped"]; totals["instances"] += len(lines)
        print(f"{split}: {len(df)} imágenes -> {img_dir}")

    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: '{dataset_dir.resolve().as_posix()}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: '{class_name}'\n",
        encoding="utf-8",
    )
    print(f"Instancias escritas: {totals['instances']} | recortadas a [0,1]: {totals['clipped']} "
          f"| descartadas: {totals['dropped']}")
    return yaml_path


def main():
    args = parse_args()
    raw_root = find_root(resolve(args.raw))
    splits_dir = resolve(args.splits)
    output_dir = resolve(args.output_dir)
    print("Datos originales:", raw_root)
    print("Partición       :", splits_dir)

    dataset_yaml = write_dataset(raw_root, splits_dir, output_dir, args.task, args.class_name, args.min_area_px)
    print("dataset.yaml    :", dataset_yaml)
    if args.prepare_only:
        return

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Falta Ultralytics. Instálalo con: python -m pip install ultralytics") from error

    model = YOLO(args.weights or DEFAULT_WEIGHTS[args.task])
    training_options = {
        "data": str(dataset_yaml),
        "task": args.task,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str((output_dir / "training").resolve()),
        "name": f"{args.task}_{args.mode}",
        "pretrained": True,
        "seed": args.seed,
        "deterministic": True,
        "single_cls": True,
        "freeze": args.freeze_layers if args.mode == "transfer" else 0,
        # aumentos coherentes con el EDA: orientación ~uniforme -> volteos válidos
        "fliplr": 0.5,
        "flipud": 0.5,
    }
    if args.device is not None:
        training_options["device"] = args.device

    model.train(**training_options)


if __name__ == "__main__":
    main()