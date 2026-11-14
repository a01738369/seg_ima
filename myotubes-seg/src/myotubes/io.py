"""Lectura de imágenes, máscaras y etiquetas YOLO (compatible con rutas de Windows)."""
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_root(base):
    """Devuelve la carpeta que contiene images/ y masks_png/ (p. ej. vt/ o vt/vt/)."""
    base = Path(base)
    candidates = [base, *sorted(p for p in base.rglob("*") if p.is_dir())]
    for p in candidates:
        if (p / "images").is_dir() and (p / "masks_png").is_dir():
            return p
    raise FileNotFoundError(f"No encontré images/ y masks_png/ dentro de {base}")


def list_files(folder, exts):
    folder = Path(folder)
    if not folder.exists():
        return {}
    return {p.stem: p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts}


def imread(path, flags=cv2.IMREAD_UNCHANGED):
    """cv2.imread que también funciona con rutas con acentos/espacios en Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"No se pudo leer {path}")
    return img


def imwrite(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"No se pudo escribir {path}")
    buf.tofile(str(path))


def read_gray(path):
    img = imread(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    return img


def read_mask(path):
    """Máscara binaria uint8 con valores {0, 1}."""
    m = imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def resize(img, scale, is_mask=False):
    if scale == 1:
        return img
    h, w = img.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # Máscaras: vecino más cercano para conservar clases discretas (sección 4.6 del documento)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, size, interpolation=interp)


def parse_yolo(path):
    """Lee un .txt YOLO-Seg. Devuelve [(clase, pts Nx2 normalizados)] y lista de problemas."""
    polys, issues = [], []
    text = Path(path).read_text().strip()
    for i, line in enumerate(text.splitlines()):
        t = line.split()
        if not t:
            continue
        try:
            cls = int(float(t[0]))
            coords = np.array(t[1:], dtype=float)
        except ValueError:
            issues.append((i, "no_numerica"))
            continue
        if len(coords) % 2:
            issues.append((i, "coords_impares"))
            coords = coords[:-1]
        pts = coords.reshape(-1, 2)
        if len(pts) < 3:
            issues.append((i, "menos_de_3_vertices"))
        if (pts < 0).any() or (pts > 1).any():
            issues.append((i, "fuera_de_rango"))
        polys.append((cls, pts))
    return polys, issues


def polygon_area_norm(pts):
    """Área (fracción de la imagen) del polígono recortado a [0,1]."""
    p = np.clip(pts, 0, 1)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
