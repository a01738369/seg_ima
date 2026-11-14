"""Pipeline de datos, línea base y métricas para segmentación de miotubos."""
import json
from pathlib import Path


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
