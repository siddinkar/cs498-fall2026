"""Resolve student tools from handout/ without installing the source tree."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts/student')]
