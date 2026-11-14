# Segmentación de miotubos — pipeline de datos y línea base

Proyecto de segmentación semántica binaria de miotubos en imágenes de microscopía (dataset `vt`).
Este repositorio contiene el análisis exploratorio, el **pipeline reproducible** que produce los
conjuntos de entrenamiento, validación y prueba a partir de los datos originales, y las **líneas
base sin deep learning** con sus métricas en validación.

## Estructura

```
myotubes-seg/
├── config.json                 # semilla, fracciones de partición, exclusiones, parámetros de la línea base
├── requirements.txt
├── scripts/
│   └── run_pipeline.py         # CLI: originales -> manifest.csv -> splits/*.csv -> sets/
├── src/myotubes/
│   ├── io.py                   # lectura de imágenes / máscaras / etiquetas YOLO
│   ├── manifest.py             # paso 1: inventario de los originales (manifest.csv)
│   ├── leakage.py              # duplicados (dHash), grupos, diagnóstico de fuga
│   ├── split.py                # paso 2: exclusiones, deduplicación, partición agrupada
│   ├── prepare.py              # paso 3: escribe sets/{train,val,test}
│   ├── baseline.py             # líneas base: Sato + umbral, y clasificador de píxeles
│   └── metrics.py              # PR-AUC, ROC-AUC, recall@precisión, matriz de confusión, IoU/Dice
├── notebooks/
│   ├── 01_EDA_miotubos.ipynb           # análisis exploratorio (con observaciones)
│   └── 02_pipeline_y_linea_base.ipynb  # pipeline, fuga, desbalance, línea base y evaluación
├── data/                       # (no versionado) raw/vt -> originales; processed -> salidas
└── reports/                    # tablas y figuras generadas para el reporte
```

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Los datos originales **no se modifican**. Copia la carpeta `vt` a `data/raw/vt` o indica su ruta
con `--raw` (CLI) o con `RAW_DIR` (notebook). Se acepta `vt/` o `vt/vt/`.

## Ejecutar el pipeline

```bash
python scripts/run_pipeline.py --config config.json --raw "C:\ruta\a\vt"
# o por pasos:
python scripts/run_pipeline.py --steps manifest split
python scripts/run_pipeline.py --steps prepare
```

| Paso | Entrada | Salida |
|---|---|---|
| 1. `manifest` | `vt/images`, `masks_png`, `labels` | `data/processed/manifest.csv`: una fila por imagen con rutas, metadatos del nombre (UUID, pozo, campo, `cellNN`), estadísticas de la máscara y las etiquetas, y dHash |
| 2. `split` | `manifest.csv` | `data/processed/splits/{train,val,test,all,descartadas,duplicate_pairs}.csv` |
| 3. `prepare` | `splits/*.csv` + originales | `data/processed/sets/<split>/{images,masks}/*.png` + `sets/<split>.csv` (escala 0.5; máscara con vecino más cercano) |

### Decisiones del paso 2 (documentadas en `config.json`)

- **Exclusiones** (se marcan con `excluded_reason`; no se borra nada):
  - máscaras vacías: mezclan negativos reales con imágenes sin anotar;
  - imágenes con una caja anotada > 20 % del área: errores de anotación.
- **Duplicados:** imágenes con distancia dHash ≤ 6 o con el mismo `cellNN` forman un clúster.
  Se conserva una copia por clúster, la de más píxeles anotados.
- **Partición agrupada:** los grupos se forman uniendo duplicados, mismo `cellNN` y mismo UUID.
  Si el UUID genera un grupo con > 35 % de las imágenes, se usa solo duplicados + `cellNN` y se
  avisa. Los grupos completos se asignan a train/val/test (70/15/15 por nº de imágenes, semilla 42).
  El script verifica que ningún grupo quede en dos particiones.
- El conjunto de **prueba** se genera pero no se usa hasta la evaluación final.

## Línea base y métricas

Ver `notebooks/02_pipeline_y_linea_base.ipynb`. Las métricas se calculan a nivel píxel en
**validación**, a escala ×0.25:

- PR-AUC (average precision), ROC-AUC;
- recall máximo con precisión ≥ 0.80;
- matriz de confusión K×K (aquí K = 2: fondo / miotubo), IoU, Dice, precisión y recall por clase.

La escala del score y el umbral de binarización se eligen **solo con train**.

## Reproducibilidad

Semilla fija (`config.json`). Las particiones quedan guardadas en `data/processed/splits/*.csv`, así
que cualquier experimento posterior debe leer esos archivos en lugar de volver a particionar.
